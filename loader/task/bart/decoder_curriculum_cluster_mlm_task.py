from loader.dataset.bart_dataset import BartDataset
from loader.task.utils.base_classifiers import BartClassifier, BartClusterClassifier
from loader.task.utils.base_cluster_mlm_task import BaseClusterMLMTask
from loader.task.utils.base_curriculum_mlm_task import BaseCurriculumMLMTask, CurriculumMLMBartBatch

from utils.transformers_adaptor import Seq2SeqModelOutput


class DecoderCurriculumClusterMLMTask(BaseCurriculumMLMTask, BaseClusterMLMTask):
    name = 'de-cu-cluster-mlm'
    mask_scheme = 'D-C-C-MASK_{de-col}'
    mask_col_ph = '{de-col}'
    dataset: BartDataset
    cls_module = BartClassifier
    cluster_cls_module = BartClusterClassifier
    batcher = CurriculumMLMBartBatch

    def __init__(
            self,
            **kwargs
    ):
        super(DecoderCurriculumClusterMLMTask, self).__init__(**kwargs)

    def init(self, **kwargs):
        super().init(**kwargs)
        self.col_pairs = [(self.p_cluster, self.p_local)]

    def _rebuild_batch(self, batch: CurriculumMLMBartBatch):
        self.prepare_batch(batch.decoder)

        self.left2right_mask(batch.decoder, self.p_cluster)

        return batch

    def produce_output(self, model_output: Seq2SeqModelOutput, **kwargs):
        kwargs['batch'] = kwargs['batch'].decoder
        return self._produce_output(model_output.last_hidden_state, **kwargs)

    def calculate_loss(self, batch: CurriculumMLMBartBatch, output, **kwargs):
        return BaseClusterMLMTask.calculate_loss(
            self,
            batch=batch.decoder,
            output=output,
            weight=batch.decoder.weight,
            **kwargs
        )

    def test__curriculum(self, batch: CurriculumMLMBartBatch, output, metric_pool):
        return BaseClusterMLMTask.test__curriculum(
            self,
            batch.decoder,
            output,
            metric_pool=metric_pool
        )
