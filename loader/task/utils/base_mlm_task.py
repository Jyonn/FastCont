import copy
from abc import ABC
from typing import Dict, Union

import numpy as np
import torch
from torch import nn

from loader.task.base_task import BaseTask, TaskLoss
from loader.task.utils.bart_classification import BartClassificationModule
from loader.task.utils.bert_classification import BertClassificationModule


class BaseMLMTask(BaseTask, ABC):
    name = 'base-mlm'
    mask_scheme = 'MASK'
    cls_module: Union[BertClassificationModule, BartClassificationModule]
    col_order: list

    def __init__(
            self,
            select_prob=0.15,
            mask_prob=0.8,
            random_prob=0.1,
            loss_pad=-100,
            apply_cols=None
    ):
        """
        :param select_prob: 选择要mask的比例
        :param mask_prob:
        :param random_prob:
        :param loss_pad:
        """
        super(BaseMLMTask, self).__init__()

        self.select_prob = select_prob
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self.loss_pad = loss_pad

        self.apply_cols = apply_cols
        self.loss_fct = nn.CrossEntropyLoss()

    def get_col_order(self, origin_order):
        if not self.apply_cols:
            return copy.deepcopy(origin_order)
        return list(filter(lambda col: col in self.apply_cols, origin_order))

    def get_expand_tokens(self):
        return [self.mask_scheme + '_' + self.dataset.COL_PH]

    def get_mask_token(self, col_name):
        return self.dataset.TOKENS[self.mask_scheme + '_' + col_name]

    def prepare_batch(self, batch):
        input_ids = batch['input_ids']  # type: torch.Tensor
        col_mask = batch['col_mask']  # type: Dict[str, torch.Tensor]
        batch_size = int(input_ids.shape[0])

        batch['mask_labels'] = torch.ones(batch_size, self.dataset.max_sequence, dtype=torch.long) * self.loss_pad
        batch['mask_labels_col'] = copy.deepcopy(col_mask)

    def do_mask(self, mask, tok, vocab_size):
        tok = int(tok)

        if np.random.uniform() < self.select_prob:
            mask_type = np.random.uniform()
            if mask_type < self.mask_prob:
                return mask, tok, True
            elif mask_type < self.mask_prob + self.random_prob:
                return np.random.randint(vocab_size), tok, False
            return tok, tok, False
        return tok, self.loss_pad, False

    def random_mask(self, batch, col_name):
        input_ids = batch['input_ids']  # type: torch.Tensor
        col_mask = batch['col_mask']  # type: Dict[str, torch.Tensor]
        mask_labels = batch['mask_labels']
        batch_size = int(input_ids.shape[0])
        vocab_size = self.depot.get_vocab_size(col_name)

        for i_batch in range(batch_size):
            for i_tok in range(self.dataset.max_sequence):
                if col_mask[col_name][i_batch][i_tok]:
                    input_id, mask_label, use_special_col = self.do_mask(
                        mask=self.get_mask_token(col_name),
                        tok=input_ids[i_batch][i_tok],
                        vocab_size=vocab_size
                    )
                    input_ids[i_batch][i_tok] = input_id
                    mask_labels[i_batch][i_tok] = mask_label
                    if use_special_col:
                        col_mask[col_name][i_batch][i_tok] = 0
                        col_mask[self.dataset.special_id][i_batch][i_tok] = 1

    def left2right_mask(self, batch, col_name):
        input_ids = batch['input_ids']  # type: torch.Tensor
        col_mask = batch['col_mask']  # type: Dict[str, torch.Tensor]
        mask_labels = batch['mask_labels']
        batch_size = int(input_ids.shape[0])

        for i_batch in range(batch_size):
            col_start, col_end = None, None
            for i_tok in range(self.dataset.max_sequence):
                if col_mask[col_name][i_batch][i_tok]:
                    if col_start is None:
                        col_start = i_tok
                    else:
                        col_end = i_tok
            col_end += 1

            if self.is_training:
                mask_count = int((col_end - col_start) * batch['mask_ratio'])
                col_start = col_end - mask_count

            selected_tokens = slice(col_start, col_end)

            mask_labels[i_batch][selected_tokens] = input_ids[i_batch][selected_tokens]
            input_ids[i_batch][selected_tokens] = self.get_mask_token(col_name)
            col_mask[col_name][i_batch][selected_tokens] = 0
            col_mask[self.dataset.special_id][i_batch][selected_tokens] = 1

    def calculate_loss(self, batch, output, **kwargs) -> TaskLoss:
        weight = kwargs.get('weight', 1)

        mask_labels_col = batch['mask_labels_col']
        mask_labels = batch['mask_labels'].to(self.device)  # type: torch.Tensor

        total_loss = torch.tensor(0, dtype=torch.float).to(self.device)
        for col_name in self.apply_cols:
            col_mask = mask_labels_col[col_name].to(self.device)  # type: torch.Tensor
            col_labels = torch.mul(col_mask, mask_labels) + \
                         torch.ones(mask_labels.shape, dtype=torch.long).to(self.device) * (col_mask - 1) * 100
            col_labels = col_labels.view(-1).to(self.device)
            vocab_size = self.depot.get_vocab_size(col_name)
            loss = self.loss_fct(
                output[col_name].view(-1, vocab_size),
                col_labels
            )
            total_loss += loss * weight
        return TaskLoss(loss=total_loss)

    def _init_extra_module(self):
        module_dict = dict()

        for col_name in self.col_order:
            vocab = self.depot.col_info[col_name].vocab

            self.print(f'preparing CLS module for {col_name} - {vocab}')
            if vocab in module_dict:
                self.print(f'exist in module dict, skip')
                continue

            if vocab in self.cls_module.classifiers:
                module_dict[vocab] = self.cls_module.classifiers[vocab]
                self.print(f'exist in cls dict module, skip')
                continue

            vocab_size = self.depot.get_vocab_size(vocab, as_vocab=True)
            module_dict[vocab] = self.cls_module(vocab, self.model_init.model_config, vocab_size)
            self.print(f'created')
        return nn.ModuleDict(module_dict)

    def _produce_output(self, last_hidden_state):
        output_dict = dict()
        for col_name in self.col_order:
            vocab = self.depot.col_info[col_name].vocab
            classification_module = self.extra_module[vocab]
            output_dict[col_name] = classification_module(last_hidden_state)
        return output_dict
