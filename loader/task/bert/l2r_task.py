import random
from typing import Dict, Optional

import torch
from torch import nn

from loader.dataset.bert_dataset import BertDataset
from loader.task.base_batch import BertBatch
from loader.task.utils.base_classifiers import BertClassifier
from utils.transformers_adaptor import BertOutput

from loader.task.base_task import BaseTask
from loader.task.base_loss import TaskLoss


class L2RBertBatch(BertBatch):
    def __init__(self, batch):
        super(L2RBertBatch, self).__init__(batch)

        self.mask_label_index = None  # type: Optional[torch.Tensor]
        self.neg_labels = None  # type: Optional[torch.Tensor]

        self.register('mask_label_index', 'neg_labels')


class L2RTask(BaseTask):
    name = 'l2r'
    dataset: BertDataset
    cls_module: BertClassifier
    batcher = L2RBertBatch

    def __init__(
            self,
            apply_col,
            neg_ratio=4,
    ):
        super(L2RTask, self).__init__()
        self.apply_col = apply_col  # type: str
        self.neg_ratio = neg_ratio

        self.loss_fct = nn.CrossEntropyLoss()

    def _rebuild_batch(self, batch: L2RBertBatch):
        mask_label_index = []
        neg_labels = []
        vocab_size = self.depot.get_vocab_size(self.apply_col)

        for i_batch in range(batch.batch_size):
            col_start = None
            col_length = 0

            for i_tok in range(self.dataset.max_sequence):
                if batch.col_mask[self.apply_col][i_batch][i_tok]:
                    if col_start is None:
                        col_start = i_tok
                    col_length += 1

            mask_index = col_length - 1 if self.test else random.choice(range(col_length))
            mask_label_index.append(mask_index + col_start - 1)

            batch.attention_mask[i_batch][mask_index + col_start: col_start + col_length] = 0

            if self.is_testing:
                continue

            neg_label = [batch.input_ids[i_batch][mask_index + col_start].item()]
            while len(neg_label) < self.neg_ratio + 1:
                neg_index = random.randint(0, vocab_size - 1)
                if neg_index != batch.input_ids[i_batch][mask_index + col_start]:
                    neg_label.append(neg_index)
            neg_labels.append(neg_label)

        batch.mask_label_index = torch.tensor(mask_label_index)
        batch.neg_labels = torch.tensor(neg_labels)
        return batch

    def _init_extra_module(self):
        vocab = self.depot.get_vocab(self.apply_col)
        vocab_size = self.depot.get_vocab_size(self.apply_col)

        self.print(f'preparing CLS module for {self.apply_col} - {vocab}')
        return self.cls_module(
            config=self.model_init.model_config,
            key=vocab,
            vocab_size=vocab_size,
        )

    def produce_output(self, model_output: BertOutput, batch):
        last_hidden_state = model_output.last_hidden_state
        return self.extra_module(last_hidden_state)

    def calculate_loss(self, batch: L2RBertBatch, output, **kwargs):
        embedding_tables = kwargs['model'].embedding_tables
        vocab_name = self.depot.get_vocab(self.apply_col)
        embeddings = embedding_tables[vocab_name]  # type: nn.Embedding

        mask_label_index = batch.mask_label_index.to(self.device)
        neg_labels = batch.neg_labels.to(self.device)  # type: torch.Tensor
        batch_size = mask_label_index.shape[0]

        if self.is_testing:
            samples = kwargs['samples']
            ranks = []
            for i_batch in range(batch_size):
                sample = samples[i_batch]['track_ids'][:-1]
                pred_embedding = output[i_batch][mask_label_index[i_batch]]
                scores = torch.mul(embeddings.weight.data, pred_embedding).mean(dim=-1)
                rank = torch.argsort(scores, descending=True).tolist()
                for item in rank:
                    if item not in sample:
                        ranks.append(item)
                        break
            return ranks

        pred_embeddings = []
        for i_batch in range(batch_size):
            pred_embeddings.append(output[i_batch][mask_label_index[i_batch]])  # [B, D]
        neg_embeddings = embeddings(neg_labels).to(self.device)  # [B, N+1, D]
        pred_embeddings = torch.stack(pred_embeddings).unsqueeze(dim=1)  # [B, 1, D]
        scores = torch.mul(neg_embeddings, pred_embeddings).mean(dim=-1)
        loss = self.loss_fct(input=scores, target=torch.tensor([0] * batch_size).to(self.device))
        return TaskLoss(loss=loss)
