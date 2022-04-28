import os

from UniTok import UniDep

from loader.bert_dataloader import BertDataLoader
from loader.bert_dataset import BertDataset
from loader.bert_init import BertInit
from loader.depot_filter import DepotFilter
from loader.embedding_init import EmbeddingInit
from loader.sample_processor import spotify_test_sample_processor
from loader.task_depot.l2r_task import L2RTask
from loader.task_depot.mlm_task import MLMTask
from loader.task_depot.non_task import NonTask
from loader.task_depot.pretrain_depot import PretrainDepot
from loader.task_depot.pretrain_task import PretrainTask
from loader.task_depot.sim_mlm_task import SimMLMTask
from utils.splitter import Splitter
from utils.time_printer import printer as print


class AtomDepot:
    def __init__(self, project_args, sub_folder=None, filter_config=None):
        data_dir = project_args.store.data_dir
        if sub_folder:
            data_dir = os.path.join(data_dir, sub_folder)

        self.depot = DepotFilter(store_dir=data_dir)
        if project_args.store.union:
            self.depot.union(*[UniDep(depot) for depot in project_args.store.union])

        if filter_config:
            print('[ATOM DEPOT] origin size:', self.depot.sample_size)
            for col, filter_list in filter_config:
                for filtering in filter_list:
                    print('[ATOM DEPOT] filtering by', filtering, 'on column', col)
                    if filtering == 'remove_empty':
                        filtering = lambda x: x
                    else:
                        filtering = eval('lambda x:' + filtering)
                    self.depot.customize(col, filtering)
                    print('[ATOM DEPOT] remaining', self.depot.sample_size, 'samples')


class AtomTask:
    def __init__(self, project_exp):
        self.tasks = []
        self.applied_task_indexes = []
        for task in project_exp.tasks:
            if task.name == 'sim-mlm':
                t = SimMLMTask
            elif task.name == 'mlm':
                t = MLMTask
            elif task.name == 'l2r':
                t = L2RTask
            elif task.name == 'non':
                t = NonTask
            else:
                raise ValueError(f'No such task: {task.name}')

            params = dict()
            if task.params:
                params = task.params.dict()
            self.tasks.append(t(**params))
            if not task.only_initialization:
                self.applied_task_indexes.append(len(self.tasks) - 1)
            print('[ATOM TASK]', task.name, 'params:', params)

        self.expand_tokens = []
        for task in self.tasks:
            self.expand_tokens.extend(task.get_expand_tokens())
        print('[ATOM TASK] expand tokens:', self.expand_tokens)


class Data:
    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'

    def __init__(self,
                 project_args,
                 project_exp,
                 device,
                 ):
        self.args = project_args
        self.exp = project_exp
        self.device = device

        atom_task = AtomTask(self.exp)
        self.tasks = atom_task.tasks

        self.depots = dict()
        self.sets = dict()

        if self.args.store.data_has_split:
            for mode, mode_config in self.args.data.split:
                filter_config = self.args.data.filter[mode]
                self.depots[mode] = AtomDepot(
                    project_args=self.args,
                    sub_folder=mode_config.path,
                    filter_config=filter_config,
                ).depot
            self.splitter = None
        else:
            filter_config = self.args.data.filter
            self.depot = AtomDepot(project_args=self.args, filter_config=filter_config).depot
            self.splitter = Splitter()
            for mode, mode_config in self.args.data.split:
                assert mode in [self.TRAIN, self.DEV, self.TEST]

                self.splitter.add(
                    name=mode,
                    weight=mode_config.weight
                )
                self.depots[mode] = self.depot

        sample_set = None
        for mode, mode_config in self.args.data.split:
            sample_pre_processor = eval(mode_config.sample_pre_processor or '0') or self.args.data.sample_pre_processor

            sample_set = BertDataset(
                depot=self.depots[mode],
                splitter=self.splitter,
                mode=mode,
                order=self.args.data.order,
                append=self.args.data.append,
                expand_tokens=atom_task.expand_tokens,
                sample_pre_processor=sample_pre_processor,
                add_sep_token=self.exp.policy.add_sep_token,
            )
            self.sets[mode] = sample_set

        self.train_set = self.sets.get(self.TRAIN)
        self.dev_set = self.sets.get(self.DEV)
        self.test_set = self.sets.get(self.TEST)

        self.embedding_init = EmbeddingInit()
        for embedding_info in self.args.embedding:
            self.embedding_init.append(**embedding_info.dict(), global_freeze=self.exp.freeze_emb)

        self.bert_init = BertInit(
            dataset=sample_set,
            embedding_init=self.embedding_init,
            global_freeze=self.exp.freeze_emb,
            **self.args.bert_config.d,
        )

        self.pretrain_depot = PretrainDepot(
            dataset=sample_set,
            bert_init=self.bert_init,
            device=self.device,
        ).register(*self.tasks)

        self.tasks = [self.tasks[index] for index in atom_task.applied_task_indexes]
        print('[DATA] after task filtering')
        for task in self.tasks:
            print('[DATA]', task.name)

    def get_loader(self, mode, *tasks: PretrainTask):
        shuffle = self.args.data.split[mode].shuffle  # NONE, FALSE, TRUE
        if shuffle not in [True, False]:  # CAN NOT USE "IF SHUFFLE"
            shuffle = self.args.data.shuffle or False

        return BertDataLoader(
            dataset=self.sets[mode],
            pretrain_tasks=list(tasks),
            shuffle=shuffle,
            batch_size=self.exp.policy.batch_size,
            pin_memory=self.exp.policy.pin_memory,
        )
