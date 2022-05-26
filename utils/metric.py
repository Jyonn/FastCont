from typing import Dict, Union, List

import torch


class Metric:
    name: str

    @classmethod
    def calculate(cls, candidates: list, candidates_set: set, ground_truth: list, n, **kwargs):
        pass


class HitRate(Metric):
    name = 'HR'

    @classmethod
    def calculate(cls, candidates: list, candidates_set: set, ground_truth: list, n, **kwargs):
        candidates_set = set(candidates[:n])
        interaction = candidates_set.intersection(set(ground_truth))
        return int(bool(interaction))


class Recall(Metric):
    name = 'RC'

    @classmethod
    def calculate(cls, candidates: list, candidates_set: set, ground_truth: list, n, **kwargs):
        candidates_set = set(candidates[:n])
        interaction = candidates_set.intersection(set(ground_truth))
        return len(interaction) * 1.0 / n


class OverlapRate(Metric):
    name = 'OR'

    @classmethod
    def calculate(cls, candidates: list, candidates_set: set, ground_truth: list, n, **kwargs):
        candidates = candidates[:n]
        candidates_set = set(candidates)
        return 1 - len(candidates_set) * 1.0 / len(candidates)


class MetricPool:
    def __init__(self):
        self.pool = []
        self.metrics = dict()  # type: Dict[str, Metric]
        self.values = dict()  # type: Dict[tuple, Union[list, float]]
        self.max_n = -1

    def add(self, *metrics: Metric, ns=None):
        ns = ns or [None]

        for metric in metrics:
            self.metrics[metric.name] = metric

            for n in ns:
                if n and n > self.max_n:
                    self.max_n = n
                self.pool.append((metric.name, n))

    def init(self):
        self.values = dict()
        for metric_name, n in self.pool:
            self.values[(metric_name, n)] = []

    def push(self, candidates, candidates_set, ground_truth, **kwargs):
        for metric_name, n in self.values:
            if n and len(ground_truth) < n:
                continue

            self.values[(metric_name, n)].append(self.metrics[metric_name].calculate(
                candidates=candidates,
                candidates_set=candidates_set,
                ground_truth=ground_truth,
                n=n,
            ))

    def export(self):
        for metric_name, n in self.values:
            self.values[(metric_name, n)] = torch.tensor(
                self.values[(metric_name, n)], dtype=torch.float).mean().item()
