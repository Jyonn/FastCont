import random
from typing import List

from torch.utils.data import DataLoader

from loader.dataset.model_dataset import ModelDataset
from loader.task.base_task import BaseTask


class ModelDataLoader(DataLoader):
    def __init__(self, dataset: ModelDataset, tasks: List[BaseTask], **kwargs):
        super().__init__(
            dataset=dataset,
            **kwargs
        )

        self.auto_dataset = dataset
        self.tasks = tasks

    def __iter__(self):
        iterator = super().__iter__()

        while True:
            try:
                batch = next(iterator)
                task = random.choice(self.tasks)
                batch = task.rebuild_batch(batch)
                batch['task'] = task
                yield batch
            except StopIteration:
                return
