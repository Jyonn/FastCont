from typing import Optional, Tuple

import torch
from torch import nn
from transformers import apply_chunking_to_forward, BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention, BertPreTrainedModel, BertIntermediate, BertOutput


class CarAttention(nn.Module):
    def __init__(self, config):
        super(CarAttention, self).__init__()
        self.self = BertSelfAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            past_key_value=None,
        )
        return self_outputs[0]


class CarLayer(nn.Module):
    def __init__(self, config):
        super(CarLayer, self).__init__()

        self.config = config
        self.attention = CarAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.linear = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def get_centroid_coherence(tril_mask, inputs):
        output = torch.matmul(tril_mask, inputs)
        row_sum = torch.sum(tril_mask, dim=-1).unsqueeze(dim=-1) + 1e-10
        mean = output / row_sum
        return torch.abs(inputs - mean)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.FloatTensor,
            tril_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        attention_output = self.attention(
            hidden_states,
            attention_mask,
        )

        # gate
        consistency = self.get_centroid_coherence(tril_mask, hidden_states)
        consistency = self.dropout(consistency)
        weights = self.linear(consistency)
        weights = self.softmax(weights)
        weights_self = weights[:, :, 0].unsqueeze(dim=-1)
        weights_attn = weights[:, :, 1].unsqueeze(dim=-1)
        attention_output = hidden_states * weights_self + attention_output * weights_attn

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class CarEncoder(nn.Module):
    def __init__(self, config):
        super(CarEncoder, self).__init__()

        self.config = config
        self.layer = nn.ModuleList([CarLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.FloatTensor,
            tril_mask: torch.Tensor,
    ) -> torch.Tensor:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                tril_mask,
            )

        return hidden_states


class CarEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))

    def forward(
            self,
            inputs_embeds
    ):
        device = inputs_embeds.device

        input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        position_ids = self.position_ids[:, :seq_length].to(device)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CarModel(BertPreTrainedModel):

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = CarEmbeddings(config)
        self.encoder = CarEncoder(config)

        self.init_weights()
        # self.post_init()

    @staticmethod
    def get_tril_mask(attention_mask: torch.Tensor):
        device = attention_mask.device
        batch_size, seq_len = attention_mask.shape
        key_mask = torch.tile(attention_mask.unsqueeze(dim=-1), [1, 1, seq_len])
        ones = torch.ones(batch_size, seq_len, seq_len).to(device)
        tril = torch.tril(ones)
        tril_mask = tril * torch.transpose(key_mask, 1, 2)
        return tril_mask

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_shape = inputs_embeds.size()[:-1]
        device = inputs_embeds.device

        tril_mask = self.get_tril_mask(attention_mask).to(device)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        embedding_output = self.embeddings(inputs_embeds)
        sequence_output = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            tril_mask=tril_mask
        )

        return sequence_output
