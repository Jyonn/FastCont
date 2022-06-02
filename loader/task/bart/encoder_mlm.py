from loader.dataset.bart_dataset import BartDataset
from loader.task.utils.base_classifiers import BartClassifier

from loader.task.utils.base_mlm_task import BaseMLMTask

from utils.transformers_adaptor import Seq2SeqModelOutput


class EncoderMLMTask(BaseMLMTask):
    name = 'en-mlm'
    mask_scheme = 'E-MASK_{en-col}'
    mask_col_ph = '{en-col}'
    dataset: BartDataset
    cls_module = BartClassifier

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init(self, **kwargs):
        super().init(**kwargs)
        self.col_order = self.get_col_order(self.dataset.encoder_order)

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
