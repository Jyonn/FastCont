import json

import torch
from torch import nn
from transformers import BertConfig


class GruConfig(BertConfig):
    def __init__(
            self,
            dropout=0.1,
            **kwargs
    ):
        super(GruConfig, self).__init__(**kwargs)
        self.dropout = dropout

    def __str__(self):
        return json.dumps(dict(
            dropout=self.dropout,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            max_position_embeddings=self.max_position_embeddings,
        ))


class GruModel(nn.Module):
    def __init__(self, config: GruConfig):
        super(GruModel, self).__init__()

        self.config = config
        self.gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            dropout=config.dropout,
        )
        self.linear = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
        )

    def forward(
            self,
            inputs_embeds: torch.Tensor,
    ):
        device = torch.get_device(inputs_embeds)
        hidden_states = torch.zeros(
            self.config.num_hidden_layers,
            self.config.max_position_embeddings,
            self.config.hidden_size,
        ).to(device)

        last_hidden_states, _ = self.gru(
            inputs_embeds,
            hidden_states
        )
        return last_hidden_states
