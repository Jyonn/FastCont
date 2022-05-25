import torch
from torch import nn
from transformers import BertConfig
from transformers.activations import ACT2FN


class BertClassificationModule(nn.Module):
    classifiers = dict()

    def __init__(self, vocab_name, config: BertConfig, vocab_size):
        super(BertClassificationModule, self).__init__()
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        self.decoder.bias = self.bias

        BertClassificationModule.classifiers[vocab_name] = self

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
