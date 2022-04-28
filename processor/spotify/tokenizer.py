import collections
import os.path
import sys
from typing import Optional

import numpy as np
import pandas as pd
from UniTok import UniTok, Column, Plot, Vocab, UniDep
from UniTok.tok import IdTok, BaseTok, EntTok, SplitTok, BertTok
from tqdm import tqdm

from tok import FollowTok, PopTok, GenresTok, DurationTok, LabelTok, FirstTok, NumTok


class Tokenizer:
    def __init__(self, data_dir, store_dir):
        self.base_dir = data_dir
        self.plist_path = os.path.join(self.base_dir, 'playlist.csv')
        # self.plist_path = os.path.join(self.base_dir, 'plist-text.csv')
        self.artist_path = os.path.join(self.base_dir, 'artist.csv')
        self.album_path = os.path.join(self.base_dir, 'album.csv')
        self.track_path = os.path.join(self.base_dir, 'track.csv')
        self.lyric_path = os.path.join(self.base_dir, 'lyric.csv')
        self.test_path = os.path.join(self.base_dir, 'challenge.csv')

        self.store_dir = store_dir
        self.plist_store = os.path.join(self.store_dir, 'plist')
        self.artist_store = os.path.join(self.store_dir, 'artist')
        self.album_store = os.path.join(self.store_dir, 'album-origin')
        self.track_store = os.path.join(self.store_dir, 'track-origin')
        self.lyric_store = os.path.join(self.store_dir, 'lyric')
        self.test_store = os.path.join(self.store_dir, 'challenge')

    def read_artist(self):
        return pd.read_csv(
            filepath_or_buffer=self.artist_path,
            sep='\t',
            header=0,
            names=['artist_id', 'artist_followers', 'artist_genres', 'artist_name', 'artist_popularity'],
        )

    def read_album(self):
        return pd.read_csv(
            filepath_or_buffer=self.album_path,
            sep='\t',
            header=0,
            names=['album_id', 'album_label', 'album_name', 'album_popularity', 'album_release_date']
        )

    def read_track(self):
        return pd.read_csv(
            filepath_or_buffer=self.track_path,
            sep='\t',
            header=0,
            names=['track_id', 'album_id', 'artist_id', 'track_duration', 'track_name', 'track_popularity']
        )

    def read_plist(self):
        return pd.read_csv(
            filepath_or_buffer=self.plist_path,
            sep='\t',
            header=0,
            names=[
                'plist_id', 'plist_name', 'plist_modify', 'plist_tracks', 'plist_albums', 'plist_edits',
                'plist_artists', 'plist_followers', 'plist_duration', 'track_ids'
            ]
        )

    def read_test(self):
        return pd.read_csv(
            filepath_or_buffer=self.test_path,
            sep='\t',
            header=0,
            names=[
                'plist_id', 'plist_name', 'plist_tracks', 'plist_holdouts', 'track_ids'
            ]
        )

    def analyse_numbers(self, type_, field):
        if type_ == 'artist':
            df = self.read_artist()
        elif type_ == 'album':
            df = self.read_album()
        elif type_ == 'track':
            df = self.read_track()
        else:
            raise ValueError

        data = df['{}_{}'.format(type_, field)].tolist()
        print(data[:10])
        print('Digit Before', len(data))
        data_ = []
        for x in data:
            try:
                data_.append(int(float(x)))
            except Exception:
                pass
        data = data_
        print('Digit After', len(data))
        data = list(map(lambda x: int(x) if isinstance(x, str) else x, data))

        counter = collections.Counter(data)
        max_count = max(counter)
        digits_max = 10
        while digits_max < max_count:
            digits_max = digits_max * 10

        bounds = []
        while digits_max >= 10:
            digits_min = digits_max // 10
            left_bound = (np.arange(9)[::-1] + 1) * digits_min
            right_bound = left_bound + digits_min
            bounds.extend(zip(left_bound, right_bound))
            digits_max = digits_min
        bounds.append((0, 1))
        bounds.reverse()

        bound_dict = dict()
        for bound in bounds:
            bound_dict[bound] = 0

        for n in tqdm(data):
            for bound in bounds:
                if bound[1] > n >= bound[0]:
                    bound_dict[bound] += 1
                    break

        print(bound_dict)

    @staticmethod
    def build_artist_tok():
        artist_tok = UniTok()
        artist_tok.add_col(Column(
            name='artist_id',
            tokenizer=IdTok(name='artist_id').as_sing()
        )).add_col(Column(
            name='artist_followers',
            tokenizer=FollowTok(name='artist_follow').as_sing(),
        )).add_col(Column(
            name='artist_genres',
            tokenizer=LabelTok(name='artist_genres', sep='@').as_list(max_length=2),
        )).add_col(Column(
            name='artist_popularity',
            tokenizer=PopTok(name='artist_popularity').as_sing(),
        ))
        return artist_tok

    @staticmethod
    def build_album_tok():
        artist_tok = UniTok()
        artist_tok.add_col(Column(
            name='album_id',
            tokenizer=IdTok(name='album_id').as_sing()
        )).add_col(Column(
            name='album_label',
            tokenizer=LabelTok(name='album_label', sep='/').as_list(max_length=2),
        )).add_col(Column(
            name='album_name',
            tokenizer=BertTok(name='bert', vocab_dir='bert-base-multilingual-cased').as_list(max_length=15),
        )).add_col(Column(
            name='album_popularity',
            tokenizer=PopTok(name='album_popularity').as_sing(),
        ))
        return artist_tok

    @staticmethod
    def build_track_tok(album_vocab: Vocab, artist_vocab: Vocab):
        track_tok = UniTok()
        track_tok.add_col(Column(
            name='track_id',
            tokenizer=IdTok(name='track_id').as_sing()
        )).add_col(Column(
            name='album_id',
            tokenizer=EntTok(name='album_id', vocab=album_vocab).as_sing()
        )).add_col(Column(
            name='artist_id',
            tokenizer=FirstTok(name='artist_ids', sep=' ', vocab=artist_vocab).as_sing()
        )).add_col(Column(
            name='track_duration',
            tokenizer=DurationTok(name='track_duration').as_sing()
        )).add_col(Column(
            name='track_popularity',
            tokenizer=PopTok(name='popularity').as_sing(),
        )).add_col(Column(
            name='track_name',
            tokenizer=BertTok(name='bert', vocab_dir='bert-base-multilingual-cased').as_list(max_length=20),
        ))
        return track_tok

    @staticmethod
    def build_challenge_tok(track_vocab):
        num_tok = NumTok(name='number')

        challenge_tok = UniTok()
        challenge_tok.add_col(Column(
            name='plist_id',
            tokenizer=IdTok(name='plist_id').as_sing()
        )).add_col(Column(
            name='plist_name',
            tokenizer=BertTok(
                name='bert',
                vocab_dir='bert-base-multilingual-cased'
            ).as_list(max_length=8)
        )).add_col(Column(
            name='track_ids',
            tokenizer=SplitTok(
                name='track_ids',
                sep=' ',
                vocab=track_vocab or Vocab(name='track_id'),
                pre_handler=lambda x: x[len('spotify:track:'):]
            ).as_list(max_length=128, slice_post=True)
        )).add_col(Column(
            name='plist_tracks',
            tokenizer=num_tok.as_sing()
        )).add_col(Column(
            name='plist_holdouts',
            tokenizer=num_tok.as_sing()
        ))
        return challenge_tok

    @staticmethod
    def build_plist_tok(track_vocab):
        plist_tok = UniTok()
        plist_tok.add_col(Column(
            name='plist_id',
            tokenizer=IdTok(name='plist_id').as_sing()
        )).add_col(Column(
            name='plist_name',
            tokenizer=BertTok(
                name='bert',
                vocab_dir='bert-base-multilingual-cased'
            ).as_list(max_length=8)
        )).add_col(Column(
            name='track_ids',
            tokenizer=SplitTok(
                name='track_ids',
                sep=' ',
                vocab=track_vocab,
                pre_handler=lambda x: x[len('spotify:track:'):]
            ).as_list(max_length=128, slice_post=True)
        ))
        return plist_tok

    @staticmethod
    def build_lyric_tok(track_vocab: Vocab):
        id_tok = IdTok(name='track_id')
        id_tok.vocab = track_vocab

        lyric_tok = UniTok()
        lyric_tok.add_col(Column(
            name='track_id',
            tokenizer=id_tok.as_sing()
        ))

    def analyse_artist(self):
        artist_tok = self.build_artist_tok()
        artist_df = self.read_artist()
        artist_tok.read_file(artist_df).analyse()

        artist_genres_vocab = artist_tok.vocab_depot('artist_genres')
        artist_genres_vocab.trim_vocab(min_frequency=5)
        artist_tok.analyse()

    def analyse_album(self):
        album_tok = self.build_album_tok()
        album_df = self.read_album()
        album_tok.read_file(album_df).analyse()

        album_label_vocab = album_tok.vocab_depot('album_label')
        album_label_vocab.trim_vocab(min_frequency=5)
        album_name_vocab = album_tok.vocab_depot('bert')
        album_name_vocab.trim_vocab(min_frequency=5)
        album_tok.analyse()

    def analyse_track(self):
        album_vocab = UniDep(self.album_store).vocab_depot('album_id')
        artist_vocab = UniDep(self.artist_store).vocab_depot('artist_id')
        track_tok = self.build_track_tok(album_vocab=album_vocab, artist_vocab=artist_vocab)
        track_df = self.read_track()
        track_tok.read_file(track_df).analyse()

    def analyse_lyric(self):
        pass

    def analyse_plist(self):
        track_vocab = UniDep(self.track_store).vocab_depot('track_id')
        track_vocab.deny_edit()
        plist_tok = self.build_plist_tok(track_vocab)
        plist_df = self.read_plist()
        plist_tok.read_file(plist_df).analyse()

    def analyse_challenge(self):
        track_vocab = UniDep(self.track_store).vocab_depot('track_id')
        track_vocab.deny_edit()
        challenge_tok = self.build_challenge_tok(track_vocab)
        challenge_df = self.read_test()
        challenge_tok.read_file(challenge_df).analyse()

    def tokenize_album(self):
        album_tok = self.build_album_tok()
        album_df = self.read_album()
        album_tok.read_file(album_df).analyse()

        album_label_vocab = album_tok.vocab_depot('album_label')
        album_label_vocab.trim_vocab(min_frequency=5)
        album_tok.tokenize().store_data(self.album_store)

    def tokenize_artist(self):
        artist_tok = self.build_artist_tok()
        artist_df = self.read_artist()
        artist_tok.read_file(artist_df).analyse()

        artist_genres_vocab = artist_tok.vocab_depot('artist_genres')
        artist_genres_vocab.trim_vocab(min_frequency=5)
        artist_tok.tokenize().store_data(self.artist_store)

    def tokenize_track(self):
        album_vocab = UniDep(self.album_store).vocab_depot('album_id')
        artist_vocab = UniDep(self.artist_store).vocab_depot('artist_id')
        track_tok = self.build_track_tok(album_vocab=album_vocab, artist_vocab=artist_vocab)
        track_df = self.read_track()
        track_tok.read_file(track_df).tokenize().store_data(self.track_store)

    def tokenize_plist(self):
        track_vocab = UniDep(self.track_store).vocab_depot('track_id')
        track_vocab.deny_edit()
        plist_tok = self.build_plist_tok(track_vocab)
        plist_df = self.read_plist()
        plist_tok.read_file(plist_df).tokenize().store_data(self.plist_store)

    def tokenize_plist_with_frequency(self):
        plist_tok = self.build_plist_tok(track_vocab=Vocab('track_id'))
        plist_df = self.read_plist()
        plist_tok.read_file(plist_df).analyse()

        track_vocab = plist_tok.vocab_depot('track_id')
        track_vocab.trim_vocab(min_frequency=10)
        plist_tok.tokenize().store_data(self.plist_store + '-fre10')

    def tokenize_challenge(self):
        # track_vocab = UniDep(self.plist_store + '-fre5').vocab_depot('track_id')
        # track_vocab.deny_edit()
        # track_vocab.frequency_mode = True
        track_vocab = None
        challenge_df = self.read_test()
        challenge_tok = self.build_challenge_tok(track_vocab=track_vocab)
        challenge_tok.read_file(challenge_df).tokenize().store_data(self.test_store)


if __name__ == '__main__':
    tokenizer = Tokenizer(
        data_dir='/data1/qijiong/Data/Spotify/format',
        store_dir='../../data/Spotify/initial',
    )

    # tokenizer.analyse_album()
    # tokenizer.tokenize_album()

    # tokenizer.analyse_artist()
    # tokenizer.tokenize_artist()

    # tokenizer.analyse_numbers('track', 'duration')
    # tokenizer.analyse_track()
    # tokenizer.tokenize_track()

    # tokenizer.analyse_plist()
    # tokenizer.tokenize_plist()
    # tokenizer.tokenize_plist_with_frequency()

    tokenizer.tokenize_challenge()
