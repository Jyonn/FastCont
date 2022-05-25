from loader.dataset.bert_dataset import BertDataset
from loader.task.utils.base_curriculum_mlm_task import BaseCurriculumMLMTask
from loader.task.utils.bert_classification import BertClassificationModule
from utils.transformers_adaptor import BertOutput


class CurriculumMLMTask(BaseCurriculumMLMTask):
    """
    MLM task for ListCont
    """

    name = 'cu-mlm'
    mask_scheme = 'MASK'
    dataset: BertDataset
    cls_module = BertClassificationModule

    def __init__(
            self,
            known_items='known_items',
            pred_items='pred_items',
            **kwargs,
    ):
        super(CurriculumMLMTask, self).__init__(**kwargs)

        self.known_items = known_items
        self.pred_items = pred_items
        self.col_order = [self.known_items, self.pred_items]

    def rebuild_batch(self, batch):
        self.prepare_batch(batch)

        if self.is_training:
            self.random_mask(batch, self.known_items)
        self.left2right_mask(batch, self.pred_items)

        return batch

    def produce_output(self, model_output: BertOutput, **kwargs):
        return self._produce_output(model_output.last_hidden_state)

    def test__hit_rate(self):
        return self.pred_items
