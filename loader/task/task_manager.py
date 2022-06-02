from loader.task.bart.decoder_curriculum_mlm import DecoderCurriculumMLMTask
from loader.task.bart.encoder_mlm import EncoderMLMTask
from loader.task.bert.bert4rec_task import Bert4RecTask
from loader.task.bert.curriculum_cluster_mlm_task import CurriculumClusterMLMTask
from loader.task.bert.curriculum_mlm_task import CurriculumMLMTask
from loader.task.non_task import NonTask
from utils.smart_printer import printer


class TaskManager:
    TASKS_LIST = [
        # Blank Task
        NonTask,

        # BERT-based
        CurriculumMLMTask,
        CurriculumClusterMLMTask,
        Bert4RecTask,

        # BART-based
        EncoderMLMTask,
        DecoderCurriculumMLMTask,
    ]

    TASKS = {task.name: task for task in TASKS_LIST}

    def __init__(self, project_exp):
        self.print = printer.ATOM__TASK_Cblue_
        self.tasks = []
        self.applied_task_indexes = []
        self.injection_task = None

        for task_config in project_exp.tasks:
            if task_config.name not in TaskManager.TASKS:
                raise ValueError(f'No matched task: {task_config.name}')

            task_class = TaskManager.TASKS[task_config.name]
            params = task_config.params.dict()

            task = task_class(**params)
            self.tasks.append(task)
            if not task_config.only_initialization:
                self.applied_task_indexes.append(len(self.tasks) - 1)
            if task.injection:
                self.injection_task = task

            self.print(task_config.name, 'params:', params)

        self.expand_tokens = []
        for task in self.tasks:
            self.expand_tokens.extend(task.get_expand_tokens())
        self.print('expand tokens:', self.expand_tokens)
