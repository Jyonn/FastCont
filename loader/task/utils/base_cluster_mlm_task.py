import json
import os.path
from abc import ABC
from typing import Union

import torch
from UniTok import Vocab
from torch import nn

from loader.task.base_loss import TaskLoss
from loader.task.utils.base_classifiers import BertClusterClassifier, BartClusterClassifier, BertClassifier, \
    BartClassifier
from loader.task.utils.base_mlm_task import BaseMLMTask, MLMBertBatch


class ClusterMLMTaskLoss(TaskLoss):
    def __init__(self, loss, cluster_loss):
        super(ClusterMLMTaskLoss, self).__init__(loss=loss)
        self.cluster_loss = cluster_loss

    def backward(self):
        loss = self.loss + self.cluster_loss
        if loss.requires_grad:
            loss.backward()


class BaseClusterMLMTask(BaseMLMTask, ABC):
    name = 'base-cluster-mlm'
    cluster_cls_module: Union[BertClusterClassifier, BartClusterClassifier]
    cls_module: Union[BertClassifier, BartClassifier]

    def __init__(
            self,
            cluster_json,
            k_global='k_global',
            p_global='p_global',
            k_local='k_local',
            p_local='p_local',
            k_cluster='k_cluster',
            p_cluster='p_cluster',
            **kwargs
    ):
        super(BaseClusterMLMTask, self).__init__(**kwargs)

        self.k_global = k_global
        self.k_local = k_local
        self.k_cluster = k_cluster
        self.p_global = p_global
        self.p_local = p_local
        self.p_cluster = p_cluster

        self.cluster_json = cluster_json
        self.col_cluster_dict = {
            self.k_global: self.k_cluster,
            self.p_global: self.p_cluster
        }

        self.col_order = [self.k_cluster, self.p_cluster]

    def _load_cluster_vocabs(self):
        vocab_path = os.path.dirname(os.path.join(self.depot.store_dir, self.cluster_json))
        return [Vocab(f'cluster_{i}').load(vocab_path) for i in range(self.n_clusters)]

    def init(self, **kwargs):
        super().init(**kwargs)

        self.cluster_vocab_count = json.load(open(os.path.join(self.depot.store_dir, self.cluster_json)))
        self.n_clusters = len(self.cluster_vocab_count)

        cluster_vocabs = self._load_cluster_vocabs()
        global_vocab = self.depot.vocab_depot(self.depot.get_vocab(self.p_global))
        self.local_global_maps = []
        for vocab in cluster_vocabs:  # type: Vocab
            map_ = []
            for index in range(vocab.get_size()):
                map_.append(global_vocab.obj2index[vocab.index2obj[index]])
            self.local_global_maps.append(map_)

    def get_embedding(self, **kwargs):
        return super().get_embedding(**kwargs, enable_attrs={self.k_global, self.p_global})

    def update_clusters(self, batch):
        # for col_name, cluster in self.col_cluster_dict.items():
        #     mask_labels = batch['mask_labels_col'][col_name]
        #     attr_labels = batch['attr_ids'][col_name][cluster]
        #     col_mask = batch['col_mask'][col_name]
        #
        #     masked_elements = torch.not_equal(col_mask, mask_labels)
        #     batch['attr_ids'][col_name][cluster] = masked_elements * (attr_labels + 1) - 1
        pass

    def _init_extra_module(self):
        return nn.ModuleDict(dict(
            cluster_cls=self.cluster_cls_module(
                cluster_vocabs=self.cluster_vocab_count,
                config=self.model_init.model_config,
            ),
            cls=self.cls_module(
                config=self.model_init.model_config,
                vocab_name=self.depot.get_vocab(self.k_cluster),
                vocab_size=self.depot.get_vocab_size(self.k_cluster)
            )
        ))

    def _produce_output(self, last_hidden_state, **kwargs):
        batch = kwargs['batch']
        mask_labels = batch['mask_labels'].to(self.device)  # type: torch.Tensor

        output_dict = dict()

        cls_module = self.extra_module['cls']
        cluster_cls_module = self.extra_module['cluster_cls']

        output_dict['pred_cluster_distribution'] = pred_clusters = cls_module(last_hidden_state)  # [B, N, V]
        output_dict['pred_cluster_labels'] = pred_cluster_labels = torch.argmax(pred_clusters.detach(), dim=-1).to(self.device)

        for col_name, local_col_name in [(self.k_cluster, self.k_local), (self.p_cluster, self.p_local)]:
            mask_labels_col = batch['mask_labels_col'][col_name].to(self.device)
            col_mask = batch['col_mask'][col_name].to(self.device)
            masked_elements = torch.not_equal(col_mask, mask_labels_col)

            if not self.is_testing:
                current_cluster_labels = masked_elements * (mask_labels + 1)
                current_pred_cluster_labels = masked_elements * (pred_cluster_labels + 1)
                cluster_labels = torch.eq(current_cluster_labels, current_pred_cluster_labels) * current_cluster_labels - 1
            else:
                cluster_labels = masked_elements * (pred_cluster_labels + 1) - 1

            output_dict[local_col_name] = cluster_cls_module(
                last_hidden_state,
                cluster_labels=cluster_labels,
            )

        return output_dict

    def _calculate_loss(self, batch: MLMBertBatch, output, **kwargs) -> TaskLoss:
        weight = kwargs.get('weight', 1)
        mask_labels = batch.mask_labels.to(self.device)  # type: torch.Tensor

        total_cluster_loss = torch.tensor(0, dtype=torch.float).to(self.device)
        total_loss = torch.tensor(0, dtype=torch.float).to(self.device)
        for col_name, local_col_name in [(self.k_cluster, self.k_local), (self.p_cluster, self.p_local)]:

            # Calculate cluster prediction loss
            vocab_size = self.depot.get_vocab_size(col_name)
            mask_labels_col = batch.mask_labels_col[col_name].to(self.device)
            col_labels = torch.mul(mask_labels_col, mask_labels) + torch.ones(
                mask_labels.shape, dtype=torch.long).to(self.device) * (1 - mask_labels_col) * self.loss_pad
            col_labels = col_labels.view(-1).to(self.device)

            loss = self.loss_fct(
                output['pred_cluster_distribution'].view(-1, vocab_size),
                col_labels
            )
            total_cluster_loss += loss * weight

            # Calculate local prediction loss

            col_mask = batch.col_mask[col_name].to(self.device)
            masked_elements = torch.not_equal(col_mask, mask_labels_col).to(self.device)  # [B, N]

            cluster_labels = masked_elements * (mask_labels + 1)
            pred_cluster_labels = masked_elements * (output['pred_cluster_labels'] + 1)
            cluster_labels = torch.eq(cluster_labels, pred_cluster_labels) * cluster_labels - 1

            for i_cluster in range(self.n_clusters):
                if not (cluster_labels == i_cluster).sum():
                    continue

                cluster_masked_elements = ((cluster_labels == i_cluster) * masked_elements).to(self.device)
                i_cluster_labels = torch.masked_select(mask_labels, cluster_masked_elements).to(self.device)  # [B, K]
                loss = self.loss_fct(
                    output[local_col_name][i_cluster],  # [B, K, V]
                    i_cluster_labels,  # [B, K]
                )
                total_loss += loss * weight / self.n_clusters
        return ClusterMLMTaskLoss(loss=total_loss, cluster_loss=total_cluster_loss)
