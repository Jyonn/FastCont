import random
from abc import ABC

from loader.task.base_task import TaskLoss
from loader.task.utils.base_mlm_task import BaseMLMTask


class BaseCurriculumMLMTask(BaseMLMTask, ABC):
    name = 'base-curriculum-mlm'

    def __init__(
            self,
            weighted=False,
            curriculum_steps=10,
            weight_decay=1,
            **kwargs
    ):
        """

        :param curriculum_steps: 课程学习总步长
        :param weighted: 是否使用加权模式（如当前过了40%的epoch，则mask 40%的数据权重为1）
        :param weight_decay: 加权模式下的权重衰减

        """
        super(BaseCurriculumMLMTask, self).__init__(**kwargs)
        self.curriculum_steps = curriculum_steps  # 10
        self.current_mask_ratio = 0

        self.weighted = weighted
        self.weight_decay = weight_decay

    def start_epoch(self, current_epoch, total_epoch):  # 3 50
        if self.weighted:
            self.current_mask_ratio = (current_epoch + 1) * 1.0 / total_epoch
        else:
            self.current_mask_ratio = \
                (int(current_epoch * self.curriculum_steps // total_epoch) + 1) * 1.0 / self.curriculum_steps
        self.print(f'set current mask ratio to {self.current_mask_ratio}')

    def prepare_batch(self, batch):
        super().prepare_batch(batch)
        if self.weighted:
            mask_ratio = random.random()
            batch['weight'] = (1 - abs(self.current_mask_ratio - mask_ratio)) * self.weight_decay
        else:
            mask_ratio = self.current_mask_ratio
            batch['weight'] = 1
        batch['mask_ratio'] = mask_ratio
        if self.is_training:
            batch['weight'] = 1

    def test__hit_rate(self):
        raise NotImplementedError

    def calculate_loss(self, batch, output, **kwargs) -> TaskLoss:
        return super(BaseCurriculumMLMTask, self).calculate_loss(
            batch=batch,
            output=output,
            weight=batch['weight'],
            **kwargs
        )
