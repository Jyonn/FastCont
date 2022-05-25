import numpy as np
import torch

from loader.dataset.model_dataset import ModelDataset


class BartDataset(ModelDataset):
    EN_COL_PH = '{en-col}'
    DE_COL_PH = '{de-col}'

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
        for col_name in order:
            if col_name in self.col_info:
                max_length = self.col_info[col_name].max_length or 1
                max_sequence += max_length + int(self.use_sep_token)  # [SEP]
            else:
                raise ValueError(f'Column [{col_name}] not exist')
        return max_sequence

    @staticmethod
    def _generate_expand_tokens(placeholder: str, token: str, col_list):
        return [token.replace(placeholder, col_name) for col_name in col_list]

    def _format_expand_tokens(self, expand_tokens):
        expand_tokens_ = []
        for token in expand_tokens or []:
            if self.EN_COL_PH in token:
                expand_tokens_.extend(self._generate_expand_tokens(self.EN_COL_PH, token, self.encoder_order))
            elif self.DE_COL_PH in token:
                expand_tokens_.extend(self._generate_expand_tokens(self.DE_COL_PH, token, self.decoder_order))
            elif self.COL_PH in token:
                expand_tokens_.extend(self._generate_expand_tokens(self.COL_PH, token, {*self.encoder_order, *self.decoder_order}))
            else:
                expand_tokens_.append(token)
        return expand_tokens_

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
        self.use_cols = list({*self.encoder_order, *self.decoder_order})

        encoder_max_sequence = self._init_max_sequence(self.encoder_order)
        decoder_max_sequence = self._init_max_sequence(self.decoder_order)
        self.encoder_token_types = len(self.encoder_order)
        self.decoder_token_types = len(self.decoder_order)

        # encoder and decoder will share max length
        self.max_sequence = max(encoder_max_sequence, decoder_max_sequence)

        expand_tokens = self._format_expand_tokens(expand_tokens)

        self.special_tokens = list(range(3 + len(expand_tokens)))
        self.PAD, self.BOS, self.SEP, *token_ids = self.special_tokens

        self.TOKENS = dict(PAD=self.PAD, BOS=self.BOS, SEP=self.SEP)
        for token, token_id in zip(expand_tokens, token_ids):
            self.TOKENS[token] = token_id

    def pad(self, sequence: list):
        return sequence + [self.PAD] * (self.max_sequence - len(sequence))

    def build_format_sequence(self, sample, order):
        col_mask = dict()
        input_ids = []
        token_type_ids = []
        special_mask = torch.tensor([1] * self.max_sequence, dtype=torch.long)
        attention_mask = torch.tensor([1] * self.max_sequence, dtype=torch.long)
        position = len(input_ids)
        token_type = 0

        for col_name in order:
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

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=token_type_ids,
            col_mask=col_mask,
        )

    def build_format_data(self, sample):
        encoder_data = self.build_format_sequence(
            sample=sample,
            # max_sequence=self.encoder_max_sequence,
            order=self.encoder_order
        )
        decoder_data = self.build_format_sequence(
            sample=sample,
            # max_sequence=self.decoder_max_sequence,
            order=self.decoder_order,
            # add_head_bos=True
        )

        append_info = dict()
        for col_name in self.append:
            append_info[col_name] = torch.tensor(sample[col_name])

        return dict(
            encoder=encoder_data,
            decoder=decoder_data,
            append_info=append_info,
        )
