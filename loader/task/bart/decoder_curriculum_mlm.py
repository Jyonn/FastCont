from loader.dataset.bart_dataset import BartDataset
from loader.task.utils.bart_classification import BartClassificationModule

from loader.task.utils.base_curriculum_mlm_task import BaseCurriculumMLMTask

from utils.transformers_adaptor import Seq2SeqModelOutput


class DecoderCurriculumMLMTask(BaseCurriculumMLMTask):
    name = 'de-cu-mlm'
    mask_scheme = 'D-C-MASK_{de-col}'
    mask_col_ph = '{de-col}'
    dataset: BartDataset
    cls_module = BartClassificationModule

    def __init__(
            self,
            **kwargs
    ):
        super(DecoderCurriculumMLMTask, self).__init__(**kwargs)

    def init(self, **kwargs):
        super().init(**kwargs)
        self.col_order = self.get_col_order(self.dataset.decoder_order)

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

    def test__curriculum(self, batch, output, metric_pool):
        mask_labels_col = batch['decoder']['mask_labels_col']
        indexes = batch['append_info']['index']
        self._test__curriculum(
            indexes=indexes,
            mask_labels_col=mask_labels_col,
            output=output,
            metric_pool=metric_pool,
            col_name=self.dataset.decoder_order[0]
        )
