import json
import os.path
from abc import ABC
from typing import Union

import torch

from loader.task.base_task import TaskLoss
from loader.task.utils.base_classifiers import BertClusterClassifier, BartClusterClassifier
from loader.task.utils.base_mlm_task import BaseMLMTask


class BaseClusterMLMTask(BaseMLMTask, ABC):
    name = 'base-cluster-mlm'
    cls_module: Union[BertClusterClassifier, BartClusterClassifier]

    def __init__(
            self,
            known_items,
            pred_items,
            known_clusters,
            pred_clusters,
            cluster_json,
            **kwargs
    ):
        super(BaseClusterMLMTask, self).__init__(**kwargs)

        self.known_items = known_items
        self.known_clusters = known_clusters
        self.pred_items = pred_items
        self.pred_clusters = pred_clusters

        self.cluster_json = cluster_json
        self.col_cluster_dict = {
            self.known_items: self.known_clusters,
            self.pred_items: self.pred_clusters
        }

    def init(self, **kwargs):
        super().init(**kwargs)
        self.cluster_vocabs = json.load(open(os.path.join(self.depot.store_dir, self.cluster_json)))

    def get_embedding(self, **kwargs):
        return super().get_embedding(**kwargs, enable_attrs=False)

    def update_clusters(self, batch):
        for col_name, cluster in self.col_cluster_dict.items():
            mask_labels = batch['mask_labels_col'][col_name]
            attr_labels = batch['attr_ids'][col_name][cluster]
            col_mask = batch['col_mask'][col_name]

            masked_elements = torch.not_equal(col_mask, mask_labels)
            batch['attr_ids'][col_name][cluster] = masked_elements * (attr_labels + 1) - 1

    def calculate_loss(self, batch, output, **kwargs) -> TaskLoss:
        weight = kwargs.get('weight', 1)

        mask_labels = batch['mask_labels'].to(self.device)  # type: torch.Tensor

        total_loss = torch.tensor(0, dtype=torch.float).to(self.device)
        for col_name, cluster in self.col_cluster_dict.items():
            mask_labels_col = batch['mask_labels_col'][col_name].to(self.device)
            col_mask = batch['col_mask'][col_name].to(self.device)
            masked_elements = torch.not_equal(col_mask, mask_labels_col).to(self.device)  # [B, N]

            attr_labels = batch['attr_ids'][col_name][cluster].to(self.device)  # [B, N]

            for i_cluster in range(len(self.cluster_vocabs)):
                if output[col_name][i_cluster] is None:
                    continue

                cluster_masked_elements = ((attr_labels == i_cluster) * masked_elements).to(self.device)
                cluster_labels = torch.masked_select(mask_labels, cluster_masked_elements).to(self.device)

                loss = self.loss_fct(
                    output[col_name][i_cluster],
                    cluster_labels
                )
                total_loss += loss * weight

        return TaskLoss(loss=total_loss)

    def _init_extra_module(self):
        return self.cls_module(
            cluster_vocabs=self.cluster_vocabs,
            config=self.model_init.model_config,
        )

    def _produce_output(self, last_hidden_state, **kwargs):
        output_dict = dict()
        batch = kwargs['batch']

        for col_name in self.col_order:
            output_dict[col_name] = self.extra_module(
                last_hidden_state,
                cluster_labels=batch['attr_ids'][col_name][self.col_cluster_dict[col_name]].to(self.device),
            )
        return output_dict
