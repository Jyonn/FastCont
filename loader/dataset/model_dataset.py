import numpy as np
from UniTok import UniDep

from torch.utils.data import Dataset

from utils.smart_printer import printer, Bracket, Color
from utils.splitter import Splitter


class ModelDataset(Dataset):
    depot: UniDep

    special_id = '__special_id'
    special_tokens: list

    injected_task: any = None
    max_sequence: int
    use_cols: list

    TOKENS: dict = {}
    COL_PH = '{col}'

    def __init__(
            self,
            depot: UniDep,
            splitter: Splitter = None,
            mode=None,
            inject_task=None,
            **kwargs,
    ):
        self.depot = depot
        self.col_info = self.depot.col_info
        self.splitter = splitter
        self.mode = mode

        self.sample_size = self.depot.sample_size
        self.injected_task = inject_task

        if splitter is None:
            self.split_range = (0, self.sample_size)
        else:
            self.split_range = splitter.divide(self.sample_size)[self.mode]
            assert splitter.contains(mode)

        self.print = printer[(self.__class__.__name__, Bracket.CLASS, Color.MAGENTA)]

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
        index += self.split_range[0]
        return self.pack_sample(index)

    def __len__(self):
        mode_range = self.split_range
        return mode_range[1] - mode_range[0]
