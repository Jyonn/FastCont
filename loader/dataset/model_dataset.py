import numpy as np
from UniTok import UniDep

from torch.utils.data import Dataset


class ModelDataset(Dataset):

    depot: UniDep

    order: list

    special_id = '__special_id'
    special_tokens: list

    max_sequence: int

    injected_task: any = None

    def inject_task(self, task):
        self.injected_task = task

    """
    get raw sample
    """

    def get_raw_sample(self, index):
        sample = self.depot[index]

        if self.injected_task:
            sample = self.injected_task.dataset_injector(sample)

        return sample

    """
    get packed sample
    """

    def pack_sample(self, index):
        sample = self.get_raw_sample(index)
        return self.build_format_data(sample)

    def get_pad_sample(self):
        return self.pack_sample(0)

    def pack_random_sample(self):
        return self.pack_sample(np.random.randint(len(self.depot)))

    def get_memory_bank(self, bank_size):
        memory_bank = []
        for _ in range(bank_size):
            memory_bank.append(self.pack_random_sample())
        return memory_bank

    def build_format_data(self, sample):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
