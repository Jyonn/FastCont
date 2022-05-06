from typing import Union

import torch

from model.auto_model import AutoModel
from utils.transformers_adaptor import BertModel, BertOutput

from loader.task.pretrain_task import PretrainTask


class AutoBert(AutoModel):
    model: BertModel

    def __init__(self, **kwargs):
        super(AutoBert, self).__init__(model_class=BertModel, **kwargs)

    def forward(self, batch, task: Union[str, PretrainTask]):
        attention_mask = batch['attention_mask'].to(self.device)  # type: torch.Tensor # [B, S]
        segment_ids = batch['segment_ids'].to(self.device)  # type: torch.Tensor # [B, S]

        if isinstance(task, str):
            task = self.pretrain_depot[task]

        input_embeds = task.get_embedding(
            batch=batch,
            table_dict=self.embedding_tables,
            embedding_size=self.hidden_size,
        )

        bert_output = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            token_type_ids=segment_ids,
            output_hidden_states=True,
            return_dict=True
        )  # type: BertOutput

        self.print('bert output device', bert_output.last_hidden_state.get_device())

        return task.produce_output(bert_output)
