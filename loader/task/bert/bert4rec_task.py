import numpy as np
import torch

from loader.dataset.bert_dataset import BertDataset
from loader.dataset.order import Order
from loader.task.utils.base_mlm_task import BaseMLMTask, MLMBertBatch
from loader.task.utils.base_classifiers import BertClassifier
from utils.transformers_adaptor import BertOutput


class Bert4RecBatch(MLMBertBatch):
    def __init__(self, batch):
        super(Bert4RecBatch, self).__init__(batch=batch)
        self.mask_index = None

        self.register('mask_index')


class Bert4RecTask(BaseMLMTask):
    """
    MLM task for ListCont
    """

    name = 'bert4rec'
    mask_scheme = 'MASK'
    dataset: BertDataset
    cls_module = BertClassifier
    batcher = Bert4RecBatch
    injection = ['train', 'dev']

    def __init__(
            self,
            known_items='known_items',
            pred_items='pred_items',
            mask_last_ratio: float = 0.1,
            **kwargs,
    ):
        super(Bert4RecTask, self).__init__(**kwargs)

        self.known_items = known_items
        self.pred_items = pred_items
        self.concat_col = 'list'
        self.col_order = [self.concat_col]

        self.mask_last_ratio = mask_last_ratio

    # rebuild sample in dataset layer by init and dataset_injector

    def init(self, **kwargs):
        super().init(**kwargs)
        self.depot.col_info[self.concat_col] = dict(vocab=self.depot.col_info[self.known_items].vocab)

    def _injector_init(self, dataset):
        # not only one dataset is required to be initialized
        dataset.order = Order([self.concat_col])

    def sample_injector(self, sample):
        sample[self.concat_col] = sample[self.known_items] + sample[self.pred_items]
        del sample[self.known_items], sample[self.pred_items]
        return sample

    def _rebuild_batch(self, batch: Bert4RecBatch):
        self.prepare_batch(batch)

        mask_last = np.random.uniform() < self.mask_last_ratio

        if mask_last or not self.is_training:
            batch_size = int(batch.input_ids.shape[0])
            mask_index = []

            for i_batch in range(batch_size):
                col_end = None
                for i_tok in range(self.dataset.max_sequence - 1, -1, -1):
                    if batch.col_mask[self.concat_col][i_batch][i_tok]:
                        col_end = i_tok
                        break
                mask_index.append(col_end)

                batch.mask_labels[i_batch][col_end] = batch.input_ids[i_batch][col_end]
                batch.input_ids[i_batch][col_end] = self.dataset.TOKENS[self.mask_scheme]
                batch.col_mask[self.concat_col][i_batch][col_end] = 0
                batch.col_mask[self.dataset.special_id][i_batch][col_end] = 1
            if self.is_testing:
                batch.mask_index = torch.tensor(mask_index)
        else:
            self.random_mask(batch, self.concat_col)

        return batch

    def produce_output(self, model_output: BertOutput, batch: Bert4RecBatch):
        return self._produce_output(model_output.last_hidden_state, batch)

    def test__left2right(self, samples, model, metric_pool, dictifier):
        ground_truths = []
        lengths = []

        argsorts = []
        for sample in samples:
            ground_truth = sample[self.pred_items]
            lengths.append(len(sample[self.pred_items]))
            sample[self.concat_col] = sample[self.known_items]
            ground_truths.append(ground_truth)
            argsorts.append([])

        for index in range(max(lengths)):
            for sample in samples:
                sample[self.concat_col].append(0)
            batch = dictifier([self.dataset.build_format_data(sample) for sample in samples])
            batch = self._rebuild_batch(batch)

            outputs = model(
                batch=batch,
                task=self,
            )[self.concat_col]

            for i_batch in range(len(samples)):
                mask_index = batch.mask_index[i_batch]

                argsort = torch.argsort(outputs[i_batch][mask_index], descending=True).cpu().tolist()[:metric_pool.max_n]
                argsorts[i_batch].append(argsort)
                samples[i_batch][self.concat_col][-1] = argsort[0]

        for i_batch, sample in enumerate(samples):
            candidates = []
            candidates_set = set()
            for depth in range(metric_pool.max_n):
                for index in range(len(sample[self.pred_items])):
                    candidates_set.add(argsorts[index][depth])
                    candidates.append(argsorts[index][depth])
                if len(candidates_set) >= metric_pool.max_n and len(candidates) >= metric_pool.max_n:
                    break

            metric_pool.push(candidates, candidates_set, ground_truths[i_batch])
