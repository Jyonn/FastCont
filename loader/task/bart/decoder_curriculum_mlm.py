from loader.dataset.bart_dataset import BartDataset
from loader.task.utils.bart_classification import BartClassificationModule

from loader.task.utils.base_curriculum_mlm_task import BaseCurriculumMLMTask

from utils.transformers_adaptor import Seq2SeqModelOutput


class DecoderCurriculumMLMTask(BaseCurriculumMLMTask):
    name = 'de-cu-mlm'
    mask_scheme = 'D-C-MASK'
    dataset: BartDataset
    cls_module = BartClassificationModule

    def __init__(
            self,
            **kwargs
    ):
        super(DecoderCurriculumMLMTask, self).__init__(**kwargs)
        self.col_order = self.get_col_order(self.dataset.decoder_order)

    def get_expand_tokens(self):
        return [self.mask_scheme + '_' + self.dataset.DE_COL_PH]

    def rebuild_batch(self, batch):
        batch_ = batch
        batch = batch['decoder']

        self.prepare_batch(batch)

        for col_name in self.col_order:
            self.left2right_mask(batch, col_name)
        return batch_

    def produce_output(self, model_output: Seq2SeqModelOutput, **kwargs):
        return self._produce_output(model_output.last_hidden_state)

    def calculate_loss(self, batch, output, **kwargs):
        batch = batch['decoder']
        return super().calculate_loss(batch, output, **kwargs)

    def test__hit_rate(self):
        return self.dataset.decoder_order[0]
