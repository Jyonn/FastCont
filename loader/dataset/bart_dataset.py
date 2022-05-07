import numpy as np
import torch

from loader.dataset.model_dataset import ModelDataset


class BartDataset(ModelDataset):

    def _format_append(self, append):
        append = append or []
        for col_name in append:
            if not self.col_info[col_name]:
                raise ValueError('{} is not a column in data'.format(col_name))
            if 'max_length' in self.col_info[col_name]:
                raise ValueError('column {} contains a list, only single-token column is allowed in append')
        return append

    def _init_max_sequence(self, order):
        max_sequence = 1
        for col_name, col_data in self.col_info:
            if col_name in order:
                max_length = col_data.max_length or 1
                max_sequence += max_length + int(self.use_sep_token)  # [SEP]
            else:
                raise ValueError(f'Column [{col_name}] not exist')
        return max_sequence

    def _format_expand_tokens(self, expand_tokens):
        expand_tokens_ = []
        for token in expand_tokens or []:
            if '{en_col}' in token:
                for col_name in self.encoder_order:
                    expand_tokens_.append(token.replace('{en_col}', col_name))
            elif '{de_col}' in token:
                for col_name in self.decoder_order:
                    expand_tokens_.append(token.replace('{de_col}', col_name))
            elif '{col}' in token:
                for col_name in [*self.encoder_order, *self.decoder_order]:
                    expand_tokens_.append(token.replace('{col}', col_name))
            else:
                expand_tokens_.append(token)
        return list(set(expand_tokens_))

    def __init__(
            self,
            encoder_order: list,
            decoder_order: list,
            append=None,
            expand_tokens=None,
            use_sep_token=True,
            **kwargs
    ):
        super(BartDataset, self).__init__(**kwargs)

        self.col_info = self.depot.col_info
        self.use_sep_token = use_sep_token

        self.encoder_order = encoder_order
        self.decoder_order = decoder_order
        self.append = self._format_append(append)

        self.encoder_max_sequence = self._init_max_sequence(self.encoder_order)
        self.decoder_max_sequence = self._init_max_sequence(self.decoder_order)
        self.encoder_token_types = len(self.encoder_order)
        self.decoder_token_types = len(self.decoder_order)

        expand_tokens = self._format_expand_tokens(expand_tokens)

        self.special_tokens = list(range(2 + len(expand_tokens)))
        self.PAD, self.SEP, *token_ids = self.special_tokens

        self.TOKENS = dict(PAD=self.PAD, SEP=self.SEP)
        for token, token_id in zip(expand_tokens, token_ids):
            self.TOKENS[token] = token_id

    def pad(self, sequence: list, max_sequence):
        return sequence + [self.PAD] * (max_sequence - len(sequence))

    def build_format_sequence(self, sample, max_sequence, order):
        col_mask = dict()
        input_ids = []
        segment_ids = []
        special_mask = torch.tensor([1] * max_sequence, dtype=torch.long)
        attention_mask = torch.tensor([1] * max_sequence, dtype=torch.long)
        position = len(input_ids)
        token_type = 0

        for col_name in order:
            feat = sample[col_name]
            if isinstance(feat, np.ndarray):
                feat = feat.tolist()
            if not isinstance(feat, list):
                feat = [feat]

            col_mask[col_name] = torch.tensor([0] * max_sequence, dtype=torch.long)
            col_mask[col_name][position: position + len(feat)] = 1
            special_mask -= col_mask[col_name]

            input_ids.extend(feat)
            position += len(feat)
            segment_ids.extend([token_type] * (len(feat) + 1))

            if self.use_sep_token:
                input_ids.append(self.SEP)
                position += 1
                token_type += 1

        attention_mask[position:] = 0
        input_ids = torch.tensor(self.pad(input_ids, max_sequence), dtype=torch.long)
        segment_ids = torch.tensor(self.pad(segment_ids, max_sequence), dtype=torch.long)
        col_mask[self.special_id] = special_mask

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            col_mask=col_mask,
        )

    def build_format_data(self, sample):
        encoder_data = self.build_format_sequence(sample, self.encoder_max_sequence, self.encoder_order)
        decoder_data = self.build_format_sequence(sample, self.decoder_max_sequence, self.decoder_order)

        append_info = dict()
        for col_name in self.append:
            append_info[col_name] = torch.tensor(sample[col_name])

        return dict(
            encoder_data=encoder_data,
            decoder_data=decoder_data,
            append_info=append_info,
        )
