import random

import torch
from torch import nn
from transformers import BertConfig
from utils.transformers_adaptor import BertOutput

from loader.task.pretrain_task import PretrainTask, TaskLoss
from utils.dictifier import Dictifier


class ClassificationModule(nn.Module):
    def __init__(self, config: BertConfig, vocab_size):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.decoder = nn.Linear(config.hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.decoder(pooled_output)
        return pooled_output


class AlignTask(PretrainTask):
    def __init__(self):
        super().__init__(name='align')

        self.loss_fct = nn.CrossEntropyLoss()
        self.aligned_key = '__aligned'

        self.dictifier = Dictifier(aggregator=torch.tensor)

    def rebuild_batch(self, batch):
        batch_size = batch['input_ids'].shape[0]
        rebuilt_batch = []
        for _ in range(batch_size):
            if random.random() > 0.5:
                batch = self.dataset.pack_random_sample()
                batch['append_info'][self.aligned_key] = 1
            else:
                batch = self.dataset.pack_random_sample_with_unaligned_views()
                batch['append_info'][self.aligned_key] = 0
            rebuilt_batch.append(batch)
        return self.dictifier(rebuilt_batch)

    def _init_extra_module(self):
        return ClassificationModule(self.model_init.model_config, 2)

    def produce_output(self, model_output: BertOutput, **kwargs):
        return model_output.last_hidden_state

    def calculate_loss(self, batch, output, **kwargs):
        total_loss = torch.tensor(0, dtype=torch.float).to(self.device)

        align_labels = batch['append_info'][self.aligned_key].to(self.device)  # type: torch.Tensor
        loss = self.loss_fct(
            self.extra_module(output),
            align_labels
        )
        return TaskLoss(loss=loss)
