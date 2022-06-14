from loader.dataset.bert_dataset import BertDataset
from loader.task.utils.base_cluster_mlm_task import BaseClusterMLMTask
from loader.task.utils.base_curriculum_mlm_task import BaseCurriculumMLMTask, CurriculumMLMBertBatch
from loader.task.utils.base_classifiers import BertClusterClassifier, BertClassifier

from utils.transformers_adaptor import BertOutput


class CurriculumClusterMLMTask(BaseCurriculumMLMTask, BaseClusterMLMTask):
    name = 'cu-cluster-mlm'
    dataset: BertDataset
    cls_module = BertClassifier
    cluster_cls_module = BertClusterClassifier
    batcher = CurriculumMLMBertBatch

    def _rebuild_batch(self, batch):
        self.prepare_batch(batch)

        if self.is_training:
            self.random_mask(batch, self.k_cluster)
        self.left2right_mask(batch, self.p_cluster)

        return batch

    def produce_output(self, model_output: BertOutput, batch: CurriculumMLMBertBatch):
        return self._produce_output(model_output.last_hidden_state, batch)

    def test__curriculum(self, batch: CurriculumMLMBertBatch, output, metric_pool):
        return BaseClusterMLMTask.test__curriculum(
            self,
            batch,
            output,
            metric_pool=metric_pool
        )

    def calculate_loss(self, batch: CurriculumMLMBertBatch, output, **kwargs):
        return BaseClusterMLMTask.calculate_loss(
            self,
            batch=batch,
            output=output,
            weight=batch.weight,
            **kwargs
        )
