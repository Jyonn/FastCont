import copy
import random
from typing import Dict

import torch
from torch import nn

from loader.dataset.bart_dataset import BartDataset
from loader.task.utils.bart_classification import BartClassificationModule

from loader.task.base_task import TaskLoss
from loader.task.utils.base_curriculum_mlm_task import BaseCurriculumTask

from utils.transformers_adaptor import Seq2SeqModelOutput


class DecoderMLMTask(BaseCurriculumTask):
    name = 'de-mlm'
    dataset: BartDataset

    def __init__(
            self,
            loss_pad=-100,
            apply_cols=None,
            **kwargs
    ):
        super(DecoderMLMTask, self).__init__(**kwargs)
        self.loss_pad = loss_pad
        self.apply_cols = apply_cols  # type: list

        self.loss_fct = nn.CrossEntropyLoss()

    def get_expand_tokens(self):
        return ['MASK_{de_col}']

    def rebuild_batch(self, batch):
        batch_ = batch
        batch = batch['decoder']
        input_ids = batch['input_ids']  # type: torch.Tensor
        col_mask = batch['col_mask']  # type: Dict[str, torch.Tensor]
        batch_size = int(input_ids.shape[0])

        mask_labels = torch.ones(batch_size, self.dataset.max_sequence, dtype=torch.long) * -100
        batch['mask_labels_col'] = copy.deepcopy(col_mask)

        if self.weighted:
            mask_ratio = random.random()
            batch['weight'] = (1 - abs(self.current_mask_ratio - mask_ratio)) * self.weight_decay
        else:
            mask_ratio = self.current_mask_ratio
            batch['weight'] = 1
        batch['mask_ratio'] = mask_ratio

        for col_name, _ in self.depot.col_info:
            if col_name not in col_mask:
                continue

            if self.apply_cols and col_name not in self.apply_cols:
                continue

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
                    mask_count = int((col_end - col_start) * mask_ratio)
                    col_start = col_end - mask_count
                else:
                    batch['weight'] = 1

                selected_tokens = slice(col_start, col_end)

                mask_labels[i_batch][selected_tokens] = input_ids[i_batch][selected_tokens]
                input_ids[i_batch][selected_tokens] = self.dataset.TOKENS[f'MASK_{col_name}']
                col_mask[col_name][i_batch][selected_tokens] = 0
                col_mask[self.dataset.special_id][i_batch][selected_tokens] = 1

        batch['mask_labels'] = mask_labels
        return batch_

    def _init_extra_module(self):
        module_dict = dict()
        for col_name in self.dataset.decoder_order:
            if self.apply_cols and col_name not in self.apply_cols:
                continue

            vocab = self.depot.col_info.d[col_name].vocab
            if vocab in module_dict:
                self.print(f'exist cls module of vocab {vocab} for column {col_name}')
                continue

            if vocab in BartClassificationModule.classifiers:
                module_dict[vocab] = BartClassificationModule.classifiers[vocab]
                self.print(f'load cls module of vocab {vocab} for col {col_name} from BartClassificationModule')
                continue

            vocab_size = self.depot.get_vocab_size(vocab, as_vocab=True)
            module_dict[vocab] = BartClassificationModule(vocab, self.model_init.model_config, vocab_size)
            self.print(f'create cls module of vocab {vocab} for column {col_name}')
        return nn.ModuleDict(module_dict)

    def produce_output(self, bart_output: Seq2SeqModelOutput, **kwargs):
        last_hidden_state = bart_output.last_hidden_state
        output_dict = dict()
        for col_name in self.dataset.decoder_order:
            if self.apply_cols and col_name not in self.apply_cols:
                continue

            vocab = self.depot.col_info.d[col_name].vocab
            classification_module = self.extra_module[vocab]
            output_dict[col_name] = classification_module(last_hidden_state)
        return output_dict

    def calculate_loss(self, batch, output, **kwargs):
        batch = batch['decoder']
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
            total_loss += loss * batch['weight']
        return TaskLoss(loss=total_loss)

    def test__hit_rate(self):
        return self.dataset.decoder_order[0]
