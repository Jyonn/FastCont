import torch
from torch import nn
from transformers import BartConfig
from transformers.activations import ACT2FN


class BartClassificationModule(nn.Module):
    classifiers = dict()

    def __init__(self, vocab_name, config: BartConfig, vocab_size):
        super(BartClassificationModule, self).__init__()
        self.transform = nn.Linear(config.d_model, config.d_model)
        self.transform_act_fn = ACT2FN[config.activation_function]
        self.LayerNorm = nn.LayerNorm(config.d_model)

        self.decoder = nn.Linear(config.d_model, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        self.decoder.bias = self.bias

        BartClassificationModule.classifiers[vocab_name] = self

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
