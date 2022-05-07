import numpy as np
import torch

from loader.dataset.model_dataset import ModelDataset


class BertDataset(ModelDataset):

    def _format_order(self, order):
        order = order or []
        if not order:
            for col_name, col_data in self.col_info:
                if col_name != self.depot.id_col:
                    order.append(col_name)
        return order

    def _format_append(self, append):
        append = append or []
        for col_name in append:
            if not self.col_info[col_name]:
                raise ValueError('{} is not a column in data'.format(col_name))
            if 'max_length' in self.col_info[col_name]:
                raise ValueError('column {} contains a list, only single-token column is allowed in append')
        return append

    def _init_max_sequence(self):
        max_sequence = 1
        for col_name, col_data in self.col_info:
            if col_name in self.order:
                max_length = col_data.max_length or 1
                max_sequence += max_length + int(self.use_sep_token)  # [SEP]
            else:
                raise ValueError(f'Column [{col_name}] not exist')
        return max_sequence

    def _format_expand_tokens(self, expand_tokens):
        expand_tokens_ = []
        for token in expand_tokens or []:
            if '{col}' in token:
                for col_name in self.order:
                    expand_tokens_.append(token.replace('{col}', col_name))
            else:
                expand_tokens_.append(token)
        return expand_tokens_

    def __init__(
            self,
            order=None,
            append=None,
            expand_tokens=None,
            use_sep_token=True,
            **kwargs
    ):
        super(BertDataset, self).__init__(**kwargs)

        self.use_sep_token = use_sep_token
        self.order = self._format_order(order)
        self.append = self._format_append(append)

        self.max_sequence = self._init_max_sequence()
        self.token_types = len(self.order) if self.use_sep_token else 1

        expand_tokens = self._format_expand_tokens(expand_tokens)
        self.special_tokens = list(range(3 + len(expand_tokens)))
        self.PAD, self.CLS, self.SEP, *token_ids = self.special_tokens

        self.TOKENS = dict(PAD=self.PAD, CLS=self.CLS, SEP=self.SEP)
        for token, token_id in zip(expand_tokens, token_ids):
            self.TOKENS[token] = token_id

    def pad(self, sequence: list):
        return sequence + [self.PAD] * (self.max_sequence - len(sequence))

    def build_format_data(self, sample):
        col_mask = dict()
        input_ids = [self.CLS]
        token_type_ids = [0]
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
            token_type_ids.extend([token_type] * (len(feat) + 1))

            if self.use_sep_token:
                input_ids.append(self.SEP)
                position += 1
                token_type += 1

        attention_mask[position:] = 0
        input_ids = torch.tensor(self.pad(input_ids), dtype=torch.long)
        token_type_ids = torch.tensor(self.pad(token_type_ids), dtype=torch.long)
        col_mask[self.special_id] = special_mask

        append_info = dict()
        for col_name in self.append:
            append_info[col_name] = torch.tensor(sample[col_name])

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=token_type_ids,
            col_mask=col_mask,
            append_info=append_info,
        )
