from loader.dataset.bert_dataset import BertDataset
from loader.task.utils.base_cluster_mlm_task import BaseClusterMLMTask
from loader.task.utils.base_curriculum_mlm_task import BaseCurriculumMLMTask
from loader.task.utils.base_classifiers import BertClusterClassifier
from utils.transformers_adaptor import BertOutput


class CurriculumClusterMLMTask(BaseCurriculumMLMTask, BaseClusterMLMTask):
    """
    MLM task for ListCont
    """

    name = 'cu-cluster-mlm'
    # mask_scheme = 'MASK'
    dataset: BertDataset
    cls_module = BertClusterClassifier

    def __init__(
            self,
            **kwargs,
    ):
        super(CurriculumClusterMLMTask, self).__init__(**kwargs)

        self.col_order = [self.known_items, self.pred_items]

    def rebuild_batch(self, batch):
        self.prepare_batch(batch)

        if self.is_training:
            self.random_mask(batch, self.known_items)
        self.left2right_mask(batch, self.pred_items)

        self.update_clusters(batch)

        return batch

    def produce_output(self, model_output: BertOutput, **kwargs):
        return self._produce_output(model_output.last_hidden_state, **kwargs)

    def test__curriculum(self, batch, output, metric_pool):
        mask_labels_col = batch['mask_labels_col']
        indexes = batch['append_info']['index']
        self._test__curriculum(
            indexes=indexes,
            mask_labels_col=mask_labels_col,
            output=output,
            metric_pool=metric_pool,
            col_name=self.pred_items,
        )
