from abc import ABC

from loader.task.base_task import BaseTask


class BaseCurriculumTask(BaseTask, ABC):
    name = 'base-curriculum'

    def __init__(
            self,
            curriculum_steps=10,
    ):
        super(BaseCurriculumTask, self).__init__()
        self.curriculum_steps = curriculum_steps  # 10
        self.current_mask_ratio = 0

    def start_epoch(self, current_epoch, total_epoch):  # 3 50
        self.current_mask_ratio = \
            (int(current_epoch * self.curriculum_steps // total_epoch) + 1) * 1.0 / self.curriculum_steps
        self.print(f'set current mask ratio to {self.current_mask_ratio}')

    def test__hit_rate(self):
        raise NotImplementedError
