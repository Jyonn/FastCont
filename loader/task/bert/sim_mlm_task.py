import copy
from typing import Dict

import numpy as np
import torch
from torch import nn
from transformers import BertConfig
from transformers.activations import ACT2FN
from utils.transformers_adaptor import BertOutput

from loader.task.pretrain_task import PretrainTask, TaskLoss
from utils.smart_printer import printer


class ClassificationModule(nn.Module):
    def __init__(self, config: BertConfig, embedding_table: nn.Embedding):
        super(ClassificationModule, self).__init__()
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.decoder = nn.Linear(config.hidden_size, vocab_size, bias=False)
        # self.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=False)
        # self.decoder.bias = self.bias
        # self.decoder.weight = embedding_table.weight
        self.embedding_table = embedding_table.weight

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)  # type: torch.Tensor  # [B, N, 768]
        # hidden_states = self.decoder(hidden_states)
        shape = list(hidden_states.shape)
        hidden_states = hidden_states.view(shape[0] * shape[1], shape[2])
        hidden_states = torch.mm(self.embedding_table.data, hidden_states.t()).t()
        shape[2] = self.embedding_table.data.shape[0]
        return hidden_states.view(*shape)
        # return hidden_states


class SimMLMTask(PretrainTask):
    def __init__(
            self,
            select_prob=0.15,
            mask_prob=0.8,
            random_prob=0.1,
            loss_pad=-100,
            only_mask_last=False,
            apply_cols=None,
    ):
        super(SimMLMTask, self).__init__(name='sim-mlm')
        self.select_prob = select_prob
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self.loss_pad = loss_pad
        self.only_mask_last = only_mask_last
        self.apply_cols = apply_cols  # type: list

        self.loss_fct = nn.CrossEntropyLoss()

        self.print = printer.SIM_MLM__TASK

    def get_expand_tokens(self):
        return ['SIM-MASK_{col}']

    def do_mask(self, mask, tok, vocab_size, is_last):
        tok = int(tok)

        if self.only_mask_last:
            if is_last:
                return mask, tok, True
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
        input_ids = batch['input_ids']  # type: torch.Tensor
        col_mask = batch['col_mask']  # type: Dict[str, torch.Tensor]
        batch_size = int(input_ids.shape[0])

        mask_labels = torch.ones(batch_size, self.dataset.max_sequence, dtype=torch.long) * -100
        batch['mask_labels_col'] = copy.deepcopy(col_mask)

        for col_name in self.depot.col_info.d:
            if col_name not in col_mask:
                continue

            if self.apply_cols and col_name not in self.apply_cols:
                continue

            vocab_size = self.depot.get_vocab_size(col_name)

            for i_batch in range(batch_size):
                for i_tok in range(self.dataset.max_sequence):
                    if col_mask[col_name][i_batch][i_tok]:
                        input_id, mask_label, use_special_col = self.do_mask(
                            mask=self.dataset.TOKENS[f'SIM-MASK_{col_name}'],
                            tok=input_ids[i_batch][i_tok],
                            vocab_size=vocab_size,
                            is_last=not col_mask[col_name][i_batch][i_tok + 1]
                        )
                        input_ids[i_batch][i_tok] = input_id
                        mask_labels[i_batch][i_tok] = mask_label
                        if use_special_col:
                            col_mask[col_name][i_batch][i_tok] = 0
                            col_mask[self.dataset.special_id][i_batch][i_tok] = 1
        batch['mask_labels'] = mask_labels
        return batch

    def _init_extra_module(self):
        module_dict = dict()
        embedding_tables = self.model_init.get_embedding_tables()

        for col_name in self.dataset.order:
            if self.apply_cols and col_name not in self.apply_cols:
                continue

            vocab = self.depot.col_info.d[col_name].vocab
            if vocab in module_dict:
                self.print('Escape create modules for', col_name, '(', vocab, ')')
                continue

            module_dict[vocab] = ClassificationModule(self.model_init.model_config, embedding_tables[vocab])
            self.print('Classification Module for', col_name, '(', vocab, ')')
        return nn.ModuleDict(module_dict)

    def produce_output(self, model_output: BertOutput, **kwargs):
        last_hidden_state = model_output.last_hidden_state

        if self.only_mask_last:
            return last_hidden_state

        output_dict = dict()
        for col_name in self.dataset.order:
            if self.apply_cols and col_name not in self.apply_cols:
                continue

            vocab = self.depot.col_info.d[col_name].vocab
            classification_module = self.extra_module[vocab]
            output_dict[col_name] = classification_module(last_hidden_state)
        return output_dict

    def calculate_loss(self, batch, output, **kwargs):
        mask_labels_col = batch['mask_labels_col']
        mask_labels = batch['mask_labels'].to(self.device)  # type: torch.Tensor

        total_loss = torch.tensor(0, dtype=torch.float).to(self.device)
        for col_name in mask_labels_col:
            if self.apply_cols and col_name not in self.apply_cols:
                continue

            if col_name == self.dataset.special_id:
                continue
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
