from loader.dataset.bart_dataset import BartDataset
from loader.task.utils.bart_classification import BartClassificationModule

from loader.task.utils.base_mlm_task import BaseMLMTask

from utils.transformers_adaptor import Seq2SeqModelOutput


class EncoderMLMTask(BaseMLMTask):
    name = 'en-mlm'
    mask_scheme = 'E-MASK'
    dataset: BartDataset
    cls_module = BartClassificationModule

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init(self, **kwargs):
        super().init(**kwargs)
        self.col_order = self.get_col_order(self.dataset.encoder_order)

    def get_expand_tokens(self):
        return [self.mask_scheme + '_{en-col}']

    def rebuild_batch(self, batch):
        batch_ = batch
        batch = batch['encoder']

        self.prepare_batch(batch)

        for col_name in self.col_order:
            self.random_mask(batch, col_name)

        return batch_

    def produce_output(self, model_output: Seq2SeqModelOutput, **kwargs):
        return self._produce_output(model_output.encoder_last_hidden_state)

    def calculate_loss(self, batch, output, **kwargs):
        batch = batch['encoder']
        return super().calculate_loss(batch, output, **kwargs)
