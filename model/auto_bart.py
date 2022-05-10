from typing import Union

import torch
from transformers.models.bart import BartModel

from loader.task.base_task import BaseTask
from model.auto_model import AutoModel


class AutoBart(AutoModel):
    model: BartModel

    def __init__(self, **kwargs):
        super(AutoBart, self).__init__(model_class=BartModel, **kwargs)

    def forward(self, batch, task: Union[str, BaseTask]):
        if isinstance(task, str):
            task = self.task_initializer[task]

        encoder_attention_mask = batch['encoder']['attention_mask'].to(self.device)  # type: torch.Tensor # [B, S]
        decoder_attention_mask = batch['decoder']['attention_mask'].to(self.device)  # type: torch.LongTensor # [B, S]

        encoder_input_embeds = task.get_embedding(
            batch=batch['encoder'],
            table_dict=self.embedding_tables,
            embedding_size=self.hidden_size,
            # input_ids_key='encoder_input_ids',
            # col_mask_key='encoder_col_mask',
        )

        decoder_input_embeds = task.get_embedding(
            batch=batch['decoder'],
            table_dict=self.embedding_tables,
            embedding_size=self.hidden_size,
            # input_ids_key='decoder_input_ids',
            # col_mask_key='decoder_col_mask',
        )

        bart_output = self.model(
            inputs_embeds=encoder_input_embeds,
            decoder_inputs_embeds=decoder_input_embeds,
            attention_mask=encoder_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        return task.produce_output(bart_output)