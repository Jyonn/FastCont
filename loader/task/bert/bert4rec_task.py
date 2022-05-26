from typing import Dict

import numpy as np
import torch

from loader.dataset.bert_dataset import BertDataset
from loader.task.utils.base_mlm_task import BaseMLMTask
from loader.task.utils.bert_classification import BertClassificationModule
from utils.transformers_adaptor import BertOutput


class Bert4RecTask(BaseMLMTask):
    """
    MLM task for ListCont
    """

    name = 'bert4rec'
    mask_scheme = 'MASK'
    dataset: BertDataset
    cls_module = BertClassificationModule

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

    def get_expand_tokens(self):
        return [self.mask_scheme]

    # rebuild sample in dataset layer by init and dataset_injector

    def init(self, **kwargs):
        super().init(**kwargs)
        self.dataset.order = [self.concat_col]

    def dataset_injector(self, sample):
        sample[self.concat_col] = sample[self.known_items] + sample[self.pred_items]
        del sample[self.known_items], sample[self.pred_items]
        return sample

    def rebuild_batch(self, batch):
        self.prepare_batch(batch)

        mask_last = np.random.uniform() < self.mask_last_ratio

        if mask_last or not self.is_training:
            input_ids = batch['input_ids']  # type: torch.Tensor
            col_mask = batch['col_mask']  # type: Dict[str, torch.Tensor]
            mask_labels = batch['mask_labels']
            batch_size = int(input_ids.shape[0])
            mask_index = []

            for i_batch in range(batch_size):
                col_end = None
                for i_tok in range(self.dataset.max_sequence - 1, -1, -1):
                    if col_mask[self.concat_col][i_batch][i_tok]:
                        col_end = i_tok
                        break
                mask_index.append(col_end)

                mask_labels[i_batch][col_end] = input_ids[i_batch][col_end]
                input_ids[i_batch][col_end] = self.dataset.TOKENS['MASK']
                col_mask[self.concat_col][i_batch][col_end] = 0
                col_mask[self.dataset.special_id][i_batch][col_end] = 1
            if not self.is_training:
                batch['mask_index'] = torch.tensor(mask_index)
        else:
            self.random_mask(batch, self.concat_col)

        return batch

    def produce_output(self, model_output: BertOutput, **kwargs):
        return self._produce_output(model_output.last_hidden_state)

    def test__left2right(self, sample, model, metric_pool):
        from utils.dictifier import Dictifier
        if not getattr(self, 'dictifier', None):
            self.dictifier = Dictifier(aggregator=torch.tensor)

        ground_truth = sample[self.pred_items]
        argsorts = []

        sample[self.concat_col] = sample[self.known_items]
        del sample[self.known_items], sample[self.pred_items]

        for index in range(len(sample[self.pred_items])):
            sample[self.concat_col].append(0)
            sample = self.dataset.build_format_data(sample)
            batch = self.dictifier([sample])

            output = model(
                batch=batch,
                task=self,
            )[self.concat_col][0]
            mask_index = batch['mask_index'][0]

            argsort = torch.argsort(output[mask_index], descending=True).cpu().tolist()[:metric_pool.max_n]
            argsorts.append(argsort)
            sample[self.concat_col][-1] = argsort[0]

        candidates = []
        candidates_set = set()
        for depth in range(metric_pool.max_n):
            for index in range(len(sample[self.pred_items])):
                candidates_set.add(argsorts[index][depth])
                candidates.append(argsorts[index][depth])
            if len(candidates_set) >= metric_pool.max_n and len(candidates) >= metric_pool.max_n:
                break

        metric_pool.push(candidates, candidates_set, ground_truth)
