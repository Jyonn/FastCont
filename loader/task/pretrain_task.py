from typing import Dict, Optional

import torch
from UniTok import UniDep
from torch import nn

from loader.dataset.model_dataset import ModelDataset
from loader.init.model_init import ModelInit


class TaskLoss:
    def __init__(self, loss: torch.Tensor):
        self.loss = loss

    def backward(self):
        self.loss.backward()


class PretrainTask:
    def __init__(self, name):
        self.name = name
        self.dataset = None  # type: Optional[ModelDataset]
        self.depot = None  # type: Optional[UniDep]
        self.model_init = None  # type: Optional[ModelInit]
        self.device = None

        self._extra_module = None  # type: Optional[nn.ModuleDict]
        self.is_training = True
        self.is_validating = False
        self.is_testing = False

    """
    Display
    """

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    """
    Task mode
    """

    def test(self):
        self.is_testing = True
        self.is_training = self.is_validating = False

    def eval(self):
        self.is_validating = True
        self.is_training = self.is_testing = False

    def train(self):
        self.is_training = True
        self.is_validating = self.is_testing = False

    def start_epoch(self, current_epoch, total_epoch):
        return

    """
    Init
    """

    @staticmethod
    def get_expand_tokens():
        return []

    def init(self, dataset: ModelDataset, model_init: ModelInit, device):
        self.dataset = dataset
        self.depot = dataset.depot
        self.model_init = model_init
        self.device = device

    """
    Extra module
    """

    @property
    def extra_module(self):
        if self._extra_module:
            return self._extra_module
        self._extra_module = self.init_extra_module()
        return self._extra_module

    def init_extra_module(self):
        raise NotImplementedError

    def rebuild_batch(self, batch):
        raise NotImplementedError

    """
    Inject dataset
    """

    # noinspection PyMethodMayBeStatic
    def dataset_injector(self, sample):
        return sample

    """
    Embedding
    """

    # noinspection PyMethodMayBeStatic
    def _get_special_seg_embedding(self, matrix: torch.Tensor, table: nn.Embedding):
        return table(matrix)

    # noinspection PyMethodMayBeStatic
    def _get_seg_embedding(self, matrix: torch.Tensor, table: nn.Embedding):
        return table(matrix)

    def get_embedding(
        self,
        batch,
        table_dict: Dict[str, nn.Embedding],
        embedding_size: int,
        input_ids_key='input_ids',
        col_mask_key='col_mask'
    ):
        input_ids = batch[input_ids_key].to(self.device)  # type: torch.Tensor
        input_embeds = torch.zeros(*input_ids.shape, embedding_size, dtype=torch.float).to(self.device)

        for col_name in batch[col_mask_key]:
            col_mask = batch[col_mask_key][col_name].to(self.device)  # type: torch.Tensor
            matrix = torch.mul(input_ids, col_mask)

            if col_name == self.dataset.special_id:
                table = table_dict[col_name]
                seg_embedding = self._get_special_seg_embedding(matrix, table).to(self.device)
            else:
                vocab = self.depot.col_info.d[col_name].vocab
                table = table_dict[vocab]
                seg_embedding = self._get_seg_embedding(matrix, table).to(self.device)
            col_mask = col_mask.unsqueeze(-1).repeat(1, 1, embedding_size).to(self.device)
            input_embeds += torch.mul(col_mask.float(), seg_embedding)

        return input_embeds

    def produce_output(self, model_output, **kwargs):
        raise NotImplementedError

    def calculate_loss(self, batch, output, **kwargs) -> TaskLoss:
        raise NotImplementedError
