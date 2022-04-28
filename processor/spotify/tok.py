import math
from typing import Optional

import pandas as pd
from UniTok import Vocab
from UniTok.tok import BaseTok, BertTok


class BoundTok(BaseTok):
    def __init__(self, name: str, start, end, vocab: Optional[Vocab] = None):
        super(BoundTok, self).__init__(name, vocab)
        self.bounds = self.get_bounds(start, end)
        self.build_vocab()

    @staticmethod
    def get_bounds(bound_start, bound_end):
        bounds = [0]
        base = bound_start

        while bound_start <= bound_end:
            bounds.append(bound_start)
            if bound_start % (base * 10) == 0:
                base *= 10
            bound_start += base
        return bounds

    def build_vocab(self):
        for i in range(len(self.bounds) - 1):
            self.vocab.append('{}-{}'.format(self.bounds[i], self.bounds[i + 1]))
        self.vocab.append('{}+'.format(self.bounds[-1]))

    def t(self, obj):
        obj = float(obj)
        if pd.isnull(obj):
            return self.vocab.append('NAN')
        obj = int(obj)
        for i in range(len(self.bounds) - 1):
            if self.bounds[i] <= obj < self.bounds[i + 1]:
                return self.vocab.append('{}-{}'.format(self.bounds[i], self.bounds[i + 1]))
        return self.vocab.append('{}+'.format(self.bounds[-1]))


class FollowTok(BoundTok):
    def __init__(self, name: str):
        super(FollowTok, self).__init__(name, start=10, end=1000000)


class PopTok(BoundTok):
    def __init__(self, name: str):
        super(PopTok, self).__init__(name, start=1, end=80)


class DurationTok(BoundTok):
    def __init__(self, name: str):
        super(DurationTok, self).__init__(name, start=10 * 1000, end=600 * 1000)


class GenresTok(BertTok):
    def __init__(self, name: str, sep: str, vocab_dir: str):
        super(GenresTok, self).__init__(name, vocab_dir=vocab_dir)
        self.sep = sep

    def t(self, obj):
        if pd.notnull(obj):
            return super().t(obj.replace(self.sep, ' '))
        return []


class FirstTok(BaseTok):
    def __init__(self, name: str, sep: str, vocab: Vocab = None):
        super(FirstTok, self).__init__(name, vocab=vocab)
        self.sep = sep

    def t(self, obj):
        assert pd.notnull(obj)
        labels = obj.split(self.sep)
        return self.vocab.append(labels[0])


class LabelTok(BaseTok):
    def __init__(self, name: str, sep: str):
        super(LabelTok, self).__init__(name)
        self.sep = sep

    def label_transform(self, label):
        vocab_transformer = {
            'Mcf Music Inc': 'Mcf Music Inc.',
            'Big Beat Records': 'Big Beat',
        }
        if label in vocab_transformer:
            label = vocab_transformer[label]

        return self.vocab.append(label)

    def t(self, obj):
        if pd.notnull(obj):
            labels = obj.split(self.sep)
            return list(map(self.label_transform, labels))
        return []


class NumTok(BaseTok):
    def t(self, obj):
        vocab_size = self.vocab.get_size()
        current = int(obj)
        if current >= vocab_size:
            for i in range(vocab_size, current + 1):
                self.vocab.append(i)
        return current


if __name__ == '__main__':
    tok = FollowTok(name='follower')
    print(tok.bounds)
    tok = PopTok(name='popularity')
    print(tok.bounds)
