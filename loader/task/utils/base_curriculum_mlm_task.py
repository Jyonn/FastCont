from abc import ABC

from loader.task.base_task import BaseTask


class BaseCurriculumTask(BaseTask, ABC):
    name = 'base-curriculum'

    def __init__(
            self,
            weighted=False,
            curriculum_steps=10,
            weight_decay=1,
    ):
        """

        :param curriculum_steps: 课程学习总步长
        :param weighted: 是否使用加权模式（如当前过了40%的epoch，则mask 40%的数据权重为1）
        :param weight_decay: 加权模式下的权重衰减
        """
        super(BaseCurriculumTask, self).__init__()
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

    def test__hit_rate(self):
        raise NotImplementedError
