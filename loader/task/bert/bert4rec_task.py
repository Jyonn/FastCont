import copy
import random
from typing import Dict

import numpy as np
import torch
from torch import nn

from loader.dataset.bert_dataset import BertDataset
from loader.task.utils.bert_classification import BertClassificationModule
from utils.transformers_adaptor import BertOutput

from loader.task.base_task import TaskLoss, BaseTask


class Bert4RecTask(BaseTask):
    """
    Bert4Rec task for ListCont
    """

    name = 'bert4rec'
    dataset: BertDataset

    def __init__(
            self,
            select_prob=0.15,
            mask_prob=0.8,
            random_prob=0.1,
            loss_pad=-100,
    ):
        super(Bert4RecTask, self).__init__()
        self.select_prob = select_prob
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self.loss_pad = loss_pad

        self.known_items = 'known_items'
        # self.pred_items = 'pred_items'
        # self.apply_cols = [self.known_items, self.pred_items]
        self.apply_cols = [self.known_items]

        self.loss_fct = nn.CrossEntropyLoss()

    def get_expand_tokens(self):
        return ['MASK_{col}']

    def do_mask(self, mask, tok, vocab_size, force_mask=None):
        tok = int(tok)

        if force_mask is True:
            return mask, tok, True
        if force_mask is False:
            return tok, self.loss_pad, False

        if np.random.uniform() < self.select_prob:
            mask_type = np.random.uniform()
            if mask_type < self.mask_prob:
                return mask, tok, True
            elif mask_type < self.mask_prob + self.random_prob:
                return np.random.randint(vocab_size), tok, False
            return tok, tok, False
        return tok, self.loss_pad, False

    def rebuild_batch(self, batch):
        assert isinstance(self.dataset, BertDataset)

        input_ids = batch['input_ids']  # type: torch.Tensor
        col_mask = batch['col_mask']  # type: Dict[str, torch.Tensor]
        batch_size = int(input_ids.shape[0])

        mask_labels = torch.ones(batch_size, self.dataset.max_sequence, dtype=torch.long) * -100
        batch['mask_labels_col'] = copy.deepcopy(col_mask)

        vocab_size = self.depot.get_vocab_size(self.known_items)
        for i_batch in range(batch_size):
            if self.is_training:
                for i_tok in range(self.dataset.max_sequence):
                    if col_mask[self.known_items][i_batch][i_tok]:
                        input_id, mask_label, use_special_col = self.do_mask(
                            mask=self.dataset.TOKENS[f'MASK_{self.known_items}'],
                            tok=input_ids[i_batch][i_tok],
                            vocab_size=vocab_size
                        )
                        input_ids[i_batch][i_tok] = input_id
                        mask_labels[i_batch][i_tok] = mask_label
                        if use_special_col:
                            col_mask[self.known_items][i_batch][i_tok] = 0
                            col_mask[self.dataset.special_id][i_batch][i_tok] = 1

            for i_tok in range(self.dataset.max_sequence):
                if col_mask[self.pred_items][i_batch][i_tok]:
                    force_mask = random.random() < self.current_mask_ratio or not self.is_training

                    input_id, mask_label, use_special_col = self.do_mask(
                        mask=self.dataset.TOKENS[f'MASK_{self.pred_items}'],
                        tok=input_ids[i_batch][i_tok],
                        vocab_size=vocab_size,
                        force_mask=force_mask,
                    )
                    input_ids[i_batch][i_tok] = input_id
                    mask_labels[i_batch][i_tok] = mask_label
                    if use_special_col:
                        col_mask[self.pred_items][i_batch][i_tok] = 0
                        col_mask[self.dataset.special_id][i_batch][i_tok] = 1

        batch['mask_labels'] = mask_labels
        return batch

    def _init_extra_module(self):
        module_dict = dict()
        for col_name in self.apply_cols:
            vocab = self.depot.col_info.d[col_name].vocab
            if vocab in module_dict:
                self.print('Escape create modules for', col_name, '(', vocab, ')')
                continue
            vocab_size = self.depot.get_vocab_size(vocab, as_vocab=True)
            module_dict[vocab] = BertClassificationModule(vocab, self.model_init.model_config, vocab_size)
            self.print('Classification Module for', col_name, '(', vocab, ')', 'with vocab size', vocab_size)
        return nn.ModuleDict(module_dict)

    def produce_output(self, model_output: BertOutput, **kwargs):
        last_hidden_state = model_output.last_hidden_state
        output_dict = dict()
        for col_name in self.apply_cols:
            vocab = self.depot.col_info.d[col_name].vocab
            classification_module = self.extra_module[vocab]
            output_dict[col_name] = classification_module(last_hidden_state)
        return output_dict

    def calculate_loss(self, batch, output, **kwargs):
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
            total_loss += loss
        return TaskLoss(loss=total_loss)

    def test__hit_rate(self):
        return self.pred_items
