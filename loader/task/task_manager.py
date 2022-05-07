from loader.task.bert.mlm_task import MLMTask
from loader.task.non_task import NonTask
from utils.smart_printer import printer


class TaskManager:
    TASKS = dict(mlm=MLMTask, non=NonTask)

    def __init__(self, project_exp):
        self.print = printer.ATOM__TASK_Cblue_
        self.tasks = []
        self.applied_task_indexes = []
        for task in project_exp.tasks:
            for cand_task in TaskManager.TASKS:
                if task.name == cand_task:
                    params = dict() if not task.params else task.params.dict()
                    self.tasks.append(TaskManager.TASKS[cand_task](**params))
                    if not task.only_initialization:
                        self.applied_task_indexes.append(len(self.tasks) - 1)
                    self.print(task.name, 'params:', params)
            else:
                raise ValueError(f'No such task: {task.name}')

        self.expand_tokens = []
        for task in self.tasks:
            self.expand_tokens.extend(task.get_expand_tokens())
        self.print('expand tokens:', self.expand_tokens)
