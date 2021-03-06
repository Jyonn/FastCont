import os
from typing import Type

from UniTok import UniDep

from loader.dataset.bart_dataset import BartDataset
from loader.dataset.bert_dataset import BertDataset
from loader.init.bart_init import BartInit
from loader.init.gru_init import GruInit
from loader.model_dataloader import ModelDataLoader
from loader.init.bert_init import BertInit
from loader.depot_manager import DepotFilter
from loader.embedding_init import EmbeddingInit

from loader.task.task_manager import TaskManager
from loader.task.task_initializer import TaskInitializer
from loader.task.base_task import BaseTask
from model.auto_bart import AutoBart
from model.auto_bert import AutoBert
from model.auto_car import AutoCar
from model.auto_caser import AutoCaser
from model.auto_gru import AutoGru
from model.auto_model import AutoModel

from utils.splitter import Splitter
from utils.smart_printer import printer


class Data:
    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'

    MODEL_BERT = 'BERT'
    MODEL_BART = 'BART'
    MODEL_CAR = 'CAR'
    MODEL_GRU = 'GRU'
    MODEL_CASER = 'CASER'

    def __init__(
        self,
        project_args,
        project_exp,
        device,
    ):
        self.args = project_args
        self.exp = project_exp
        self.device = device
        self.print = printer.DATA_Cblue_

        self.print('Model', self.exp.model)

        models = [AutoBert, AutoBart, AutoGru, AutoCar, AutoCaser]
        for model in models:  # type: Type[AutoModel]
            if self.exp.model == model.__name__[4:].upper():
                self.dataset_class = model.dataset_class
                self.model_initializer = model.model_initializer
                self.model = model
                break
        else:
            raise ValueError(f'Unknown model [{self.exp.model}]')

        self.task_manager = TaskManager(self.exp)
        self.tasks = self.task_manager.tasks

        self.depots, self.splitter = self._init_depots()

        self.sets = dict()
        for mode, mode_config in self.args.data.split:
            self.sets[mode] = self.dataset_class(
                depot=self.depots[mode],
                splitter=self.splitter,
                mode=mode,
                expand_tokens=self.task_manager.expand_tokens,
                inject_task=self.task_manager.injection_task,
                **self.args.set.d,
            )

        self.train_set = self.sets.get(self.TRAIN)
        self.dev_set = self.sets.get(self.DEV)
        self.test_set = self.sets.get(self.TEST)

        self.embedding_init = EmbeddingInit()
        for embedding_info in self.args.embedding:
            self.embedding_init.append(**embedding_info.dict(), global_freeze=self.exp.freeze_emb)

        self.model_init = self.model_initializer(
            dataset=self.train_set,
            embedding_init=self.embedding_init,
            global_freeze=self.exp.freeze_emb,
            **self.args.model_config.d,
        )

        self.task_initializer = TaskInitializer(
            dataset=self.train_set,
            model_init=self.model_init,
            device=self.device,
        ).register(*self.tasks)

        self.tasks = [self.tasks[index] for index in self.task_manager.applied_task_indexes]
        self.print('after task filtering')
        for task in self.tasks:
            self.print(task.name)

    def get_loader(self, mode, *tasks: BaseTask):
        shuffle = self.args.data.split[mode].shuffle  # NONE, FALSE, TRUE
        if shuffle not in [True, False]:  # CAN NOT USE "IF SHUFFLE"
            shuffle = self.args.data.shuffle or False

        return ModelDataLoader(
            dataset=self.sets[mode],
            tasks=list(tasks),
            shuffle=shuffle,
            batch_size=self.exp.policy.batch_size,
            pin_memory=self.exp.policy.pin_memory,
        )

    def _format_depot(self, sub_folder=None, filter_config=None):
        data_dir = self.args.store.data_dir
        if sub_folder:
            data_dir = os.path.join(data_dir, sub_folder)

        depot = DepotFilter(store_dir=data_dir)
        if self.args.store.union:
            depot.union(*[UniDep(d) for d in self.args.store.union])

        if filter_config:
            self.print.format__depot_Pcyan_('origin size:', depot.sample_size)
            for col, filter_list in filter_config:
                for filtering in filter_list:
                    self.print.format__depot_Pcyan_('filtering by', filtering, 'on column', col)
                    if filtering == 'remove_empty':
                        filtering = lambda x: x
                    else:
                        filtering = eval('lambda x:' + filtering)
                    depot.customize(col, filtering)
                    self.print.format__depot_Pcyan_('remaining', depot.sample_size, 'samples')
        return depot

    def _init_depots(self):
        depots = dict()
        splitter = None

        if self.args.store.data_has_split:
            for mode, mode_config in self.args.data.split:
                filter_config = self.args.data.filter[mode]
                depots[mode] = self._format_depot(
                    sub_folder=mode_config.path,
                    filter_config=filter_config,
                )
        else:
            filter_config = self.args.data.filter
            depot = self._format_depot(filter_config=filter_config)
            splitter = Splitter()
            for mode, mode_config in self.args.data.split:
                assert mode in [self.TRAIN, self.DEV, self.TEST]

                splitter.add(
                    name=mode,
                    weight=mode_config.weight
                )
                depots[mode] = depot

        return depots, splitter
