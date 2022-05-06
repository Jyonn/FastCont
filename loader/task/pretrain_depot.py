from typing import Dict

from torch import nn

from loader.dataset.model_dataset import ModelDataset
from loader.init.model_init import ModelInit
from loader.task_depot.pretrain_task import PretrainTask
from utils.smart_printer import printer


class PretrainDepot:
    def __init__(self, dataset: ModelDataset, model_init: ModelInit, device):
        self.print = printer.PT__DEPOT
        self.dataset = dataset
        self.bert_init = model_init
        self.device = device

        self.depot = dict()  # type: Dict[str, PretrainTask]

    def register(self, *tasks: PretrainTask):
        for task in tasks:
            self.depot[task.name] = task
            task.init(
                dataset=self.dataset,
                model_init=self.bert_init,
                device=self.device,
            )
        return self

    def __getitem__(self, item):
        return self.depot[item]

    def get_extra_modules(self):
        extra_modules = dict()
        self.print('create extra modules')
        for task_name in self.depot:
            extra_module = self.depot[task_name].init_extra_module()
            extra_modules[task_name] = extra_module
        return nn.ModuleDict(extra_modules)
