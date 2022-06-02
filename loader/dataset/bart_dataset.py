from loader.dataset.model_dataset import ModelDataset
from loader.dataset.order import Order


class BartDataset(ModelDataset):
    EN_COL_PH = '{en-col}'
    DE_COL_PH = '{de-col}'

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
            **kwargs
    ):
        super(BartDataset, self).__init__(**kwargs)

        self.encoder_order = Order(encoder_order)
        self.decoder_order = Order(decoder_order)
        self.use_cols = list({
            *self._format_use_cols(self.encoder_order),
            *self._format_use_cols(self.decoder_order)
        })

        encoder_max_sequence = self._init_max_sequence(self.encoder_order)
        decoder_max_sequence = self._init_max_sequence(self.decoder_order)
        self.encoder_token_types = len(self.encoder_order) if self.use_sep_token else 1
        self.decoder_token_types = len(self.decoder_order) if self.use_sep_token else 1

        # encoder and decoder will share max length
        self.max_sequence = max(encoder_max_sequence, decoder_max_sequence)

        self.init()

    def _build_format_data(self, sample):
        encoder_data = self.build_format_sequence(
            sample=sample,
            order=self.encoder_order
        )
        decoder_data = self.build_format_sequence(
            sample=sample,
            order=self.decoder_order,
        )

        return dict(
            encoder=encoder_data,
            decoder=decoder_data,
        )
