import json

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig


class CaserConfig(BertConfig):
    def __init__(
            self,
            num_vertical,
            num_horizontal,
            dropout=0.1,
            **kwargs,
    ):
        super(CaserConfig, self).__init__(**kwargs)
        self.num_vertical = num_vertical
        self.num_horizontal = num_horizontal
        self.dropout = dropout

    def __str__(self):
        return json.dumps(dict(
            num_vertical=self.num_vertical,
            num_horizontal=self.num_horizontal,
            dropout=self.dropout,
            hidden_size=self.hidden_size,
            max_length=self.max_length,
        ))


class CaserModel(nn.Module):
    def __init__(self, config: CaserConfig):
        super(CaserModel, self).__init__()

        self.config = config

        # vertical and horizontal conv
        self.vertical_conv = nn.Conv2d(
            in_channels=1,
            out_channels=config.num_vertical,
            kernel_size=(config.max_length, 1)
        )
        lengths = [i + 1 for i in range(config.max_length)]
        self.horizontal_conv = nn.ModuleList([nn.Conv2d(
            in_channels=1,
            out_channels=config.num_horizontal,
            kernel_size=(i, config.hidden_size)
        ) for i in lengths])

        self.fc_vertical_size = config.num_vertical * config.hidden_size
        self.fc_horizontal_size = config.num_horizontal * config.max_length
        self.fc = nn.Linear(
            in_features=self.fc_vertical_size + self.fc_horizontal_size,
            out_features=config.hidden_size,
        )

        self.dropout = nn.Dropout(config.dropout)
        self.conv_act = self.fc_act = nn.ReLU()

    def forward(
            self,
            inputs_embeds: torch.Tensor,
    ):
        inputs_embeds = inputs_embeds.unsqueeze(dim=1)
        vertical_output = self.vertical_conv(inputs_embeds)
        vertical_output = vertical_output.view(-1, self.fc_vertical_size)

        horizontal_outputs = []
        for conv in self.horizontal_conv:
            conv_output = self.conv_act(conv(inputs_embeds).squeeze(3))
            pool_output = F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2)
            horizontal_outputs.append(pool_output)
        horizontal_output = torch.cat(horizontal_outputs, dim=1)

        output = torch.cat([vertical_output, horizontal_output], 1)
        output = self.dropout(output)

        fc_output = self.fc_act(self.fc(output))

        return fc_output
