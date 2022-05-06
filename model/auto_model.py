from typing import Union

from torch import nn
from transformers import PreTrainedModel

from loader.init.model_init import ModelInit

from loader.task.pretrain_depot import PretrainDepot
from loader.task.pretrain_task import PretrainTask

from utils.smart_printer import printer, Color


class AutoModel(nn.Module):
    model: PreTrainedModel

    def __init__(
            self,
            model_init: ModelInit,
            device,
            pretrain_depot: PretrainDepot,
            model_class: callable,
    ):
        super(AutoModel, self).__init__()

        self.model_init = model_init
        self.device = device
        self.pretrain_depot = pretrain_depot

        self.hidden_size = self.model_init.hidden_size
        self.print = printer[(self.__class__.__name__, '-', Color.MAGENTA)]

        self.model = model_class(self.model_init.model_config)  # use compatible code
        self.embedding_tables = self.model_init.get_embedding_tables()
        self.extra_modules = self.pretrain_depot.get_extra_modules()
        self.print('Extra Modules', self.extra_modules)

    def forward(self, batch, task: Union[str, PretrainTask]):
        raise NotImplementedError
