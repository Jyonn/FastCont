import numpy as np
import torch
from UniTok import UniDep

from loader.dataset.model_dataset import ModelDataset
from utils.splitter import Splitter


class BertDataset(ModelDataset):
    def __init__(
            self,
            depot: UniDep,
            splitter: Splitter = None,
            mode=None,
            order=None,
            append=None,
            expand_tokens=None,
            add_sep_token=True,
    ):
        self.depot = depot

        self.max_sequence = 1  # [CLS]
        self.token_types = 0

        self.add_sep_token = add_sep_token

        order = order or []
        if not isinstance(order, list):
            order = [order]
        null_order = not order

        self.col_info = depot.col_info.dict()

        for col_name in self.col_info:
            if col_name != depot.id_col:
                if null_order:
                    order.append(col_name)
                elif col_name in order:
                    max_length = self.col_info[col_name].get('max_length', 1)
                    self.max_sequence += max_length + int(add_sep_token)  # [SEP]
                    self.token_types = self.token_types + 1 if self.add_sep_token else 1
        self.order = order

        self.expand_tokens = []
        if expand_tokens:
            for token in expand_tokens:
                if '{col}' in token:
                    for col_name in self.order:
                        self.expand_tokens.append(token.replace('{col}', col_name))
                else:
                    self.expand_tokens.append(token)

        self.append = append or []
        for col_name in self.append:
            if col_name not in self.col_info:
                raise ValueError('{} is not a column in data'.format(col_name))
            if 'max_length' in self.col_info[col_name]:
                raise ValueError('column {} contains a list, only single-token column is allowed in append')

        self.special_tokens = list(range(3 + len(self.expand_tokens)))
        self.PAD, self.CLS, self.SEP, *token_ids = self.special_tokens

        self.TOKENS = dict(PAD=self.PAD, CLS=self.CLS, SEP=self.SEP)
        for token, token_id in zip(self.expand_tokens, token_ids):
            self.TOKENS[token] = token_id

        self.mode = mode
        self.sample_size = self.depot.sample_size

        if splitter is None:
            self.split_range = (0, self.sample_size)
        else:
            self.split_range = splitter.divide(self.sample_size)[self.mode]
            assert splitter.contains(mode)

    def pad(self, sequence: list):
        return sequence + [self.PAD] * (self.max_sequence - len(sequence))

    def build_format_data(self, sample):
        col_mask = dict()
        input_ids = [self.CLS]
        segment_ids = [0]
        special_mask = torch.tensor([1] * self.max_sequence, dtype=torch.long)
        attention_mask = torch.tensor([1] * self.max_sequence, dtype=torch.long)
        position = len(input_ids)
        token_type = 0

        for col_name in self.order:
            feat = sample[col_name]
            if isinstance(feat, np.ndarray):
                feat = feat.tolist()
            if not isinstance(feat, list):
                feat = [feat]

            col_mask[col_name] = torch.tensor([0] * self.max_sequence, dtype=torch.long)
            col_mask[col_name][position: position + len(feat)] = 1
            special_mask -= col_mask[col_name]

            input_ids.extend(feat)
            position += len(feat)
            segment_ids.extend([token_type] * (len(feat) + 1))

            if self.add_sep_token:
                input_ids.append(self.SEP)
                position += 1
                token_type += 1

        attention_mask[position:] = 0
        input_ids = torch.tensor(self.pad(input_ids), dtype=torch.long)
        segment_ids = torch.tensor(self.pad(segment_ids), dtype=torch.long)
        col_mask[self.special_id] = special_mask

        append_col = dict()
        for col_name in self.append:
            append_col[col_name] = torch.tensor(sample[col_name])

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            col_mask=col_mask,
            append_info=append_col,
        )

    def __getitem__(self, index):
        index += self.split_range[0]
        return self.pack_sample(index)

    def __len__(self):
        mode_range = self.split_range
        return mode_range[1] - mode_range[0]
