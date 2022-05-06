from transformers import BartConfig

from loader.init.model_init import ModelInit


class BartInit(ModelInit):
    def __init__(
            self,
            encoder_layers=6,
            encoder_attention_heads=12,
            decoder_layers=6,
            decoder_attention_heads=12,
            **kwargs,
    ):
        super(BartInit, self).__init__(**kwargs)
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads

    @property
    def load_model_config(self):
        return BartConfig(
            vocab_size=1,
            max_position_embeddings=self.dataset.max_sequence,
            encoder_layers=self.encoder_layers,
            encoder_ffn_dim=self.hidden_size * 4,
            encoder_attention_heads=self.encoder_attention_heads,
            decoder_layers=self.decoder_layers,
            decoder_ffn_dim=self.hidden_size * 4,
            decoder_attention_heads=self.decoder_attention_heads,
            d_model=self.hidden_size,
        )
