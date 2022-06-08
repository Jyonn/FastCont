import torch

from loader.dataset.bert_dataset import BertDataset
from loader.task.utils.base_cluster_mlm_task import BaseClusterMLMTask
from loader.task.utils.base_curriculum_mlm_task import BaseCurriculumMLMTask, CurriculumMLMBertBatch
from loader.task.utils.base_classifiers import BertClusterClassifier, BertClassifier
from utils.transformers_adaptor import BertOutput


class CurriculumClusterMLMTask(BaseCurriculumMLMTask, BaseClusterMLMTask):
    """
    MLM task for ListCont
    """

    name = 'cu-cluster-mlm'
    dataset: BertDataset
    cls_module = BertClassifier
    cluster_cls_module = BertClusterClassifier
    batcher = CurriculumMLMBertBatch

    def __init__(
            self,
            **kwargs,
    ):
        super(CurriculumClusterMLMTask, self).__init__(**kwargs)

        self.col_order = [self.k_cluster, self.p_cluster]

    def _rebuild_batch(self, batch):
        self.prepare_batch(batch)

        if self.is_training:
            self.random_mask(batch, self.k_cluster)
        self.left2right_mask(batch, self.p_cluster)

        return batch

    def produce_output(self, model_output: BertOutput, **kwargs):
        return self._produce_output(model_output.last_hidden_state, **kwargs)

    def test__curriculum(self, batch, output, metric_pool):
        mask_labels_col = batch['mask_labels_col']
        indexes = batch['append_info']['index']

        pred_cluster_labels = output['pred_cluster_labels']
        output = output[self.p_local]
        col_mask = mask_labels_col[self.p_cluster]

        cluster_indexes = [0] * self.n_clusters

        for i_batch in range(len(indexes)):
            arg_sorts = []
            for i_tok in range(self.dataset.max_sequence):
                if col_mask[i_batch][i_tok]:
                    cluster_id = pred_cluster_labels[i_batch][i_tok]
                    top_items = torch.argsort(
                        output[cluster_id][cluster_indexes[cluster_id]], descending=True
                    ).cpu().tolist()[:metric_pool.max_n]
                    top_items = [self.local_global_maps[cluster_id][item] for item in top_items]
                    arg_sorts.append(top_items)
                    cluster_indexes[cluster_id] += 1
                else:
                    arg_sorts.append(None)

            ground_truth = self.depot.pack_sample(indexes[i_batch])[self.p_global][:metric_pool.max_n]
            candidates = []
            candidates_set = set()
            for depth in range(metric_pool.max_n):
                for i_tok in range(self.dataset.max_sequence):
                    if col_mask[i_batch][i_tok]:
                        candidates_set.add(arg_sorts[i_tok][depth])
                        candidates.append(arg_sorts[i_tok][depth])
                if len(candidates_set) >= metric_pool.max_n and len(candidates) >= metric_pool.max_n:
                    break

            metric_pool.push(candidates, candidates_set, ground_truth)

        for cluster_id in range(self.n_clusters):
            if output[cluster_id] is None:
                assert not cluster_indexes[cluster_id]
            else:
                assert cluster_indexes[cluster_id] == len(output[cluster_id])
