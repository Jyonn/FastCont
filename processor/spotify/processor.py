import json
import os.path

import pandas as pd
from tqdm import tqdm

from processor.processor import Processor
from utils.dictifier import Dictifier


class SpotifyPreProcessor(Processor):
    def __init__(self, data_dir, store_dir):
        super(SpotifyPreProcessor, self).__init__(data_dir=data_dir, store_dir=store_dir)
        self.index_min = 0
        self.index_max = 1000000
        self.index_step = 1000

        self.playlists = []
        self.challenges = []
        self.tracks_set = set()
        self.tracks = []
        self.un_retrieved_tracks = dict()
        self.un_retrieved_albums = dict()

        self.dictifier = Dictifier(aggregator=list)

    def retrieve_tracks_from_file(self, filepath, retrieved_tracks):
        playlists = json.load(open(filepath, 'r'))['playlists']
        for p in playlists:
            for t in p['tracks']:
                track_id = t['track_uri'][len('spotify:track:'):]
                if track_id not in retrieved_tracks:
                    if track_id not in self.un_retrieved_tracks:
                        self.un_retrieved_tracks[track_id] = dict(
                            track_id=track_id,
                            name=t['track_name'],
                            album_id=t['album_uri'][len('spotify:album:'):],
                            artist_ids=t['artist_uri'][len('spotify:artist:'):],
                        )

    def retrieve_albums_from_file(self, filepath, retrieved_albums):
        playlists = json.load(open(filepath, 'r'))['playlists']
        for p in playlists:
            for t in p['tracks']:
                album_id = t['album_uri'][len('spotify:album:'):]
                if album_id not in retrieved_albums:
                    if album_id not in self.un_retrieved_albums:
                        self.un_retrieved_albums[album_id] = dict(
                            album_id=album_id,
                            name=t['album_name'],
                        )

    def tokenize_challenge(self):
        filepath = os.path.join(self.data_dir, 'challenge', 'challenge_set.json')
        playlists = json.load(open(filepath, 'r'))['playlists']

        for p in tqdm(playlists):
            playlist = dict(
                pid=p['pid'],
                name=p['name'] if 'name' in p else '',
                num_tracks=p['num_tracks'],
                num_holdouts=p['num_holdouts']
            )

            track_ids = []
            for t in p['tracks']:
                track_id = t['track_uri']
                track_ids.append(track_id)

            playlist['track_ids'] = ' '.join(track_ids)
            self.challenges.append(playlist)

        self.challenges = self.dictifier(self.challenges)
        cl_df = pd.DataFrame(self.challenges)
        cl_df.to_csv(os.path.join(self.store_dir, 'challenge.csv'), sep='\t', index=False)

    def tokenize_file(self, filepath):
        playlists = json.load(open(filepath, 'r'))['playlists']
        for p in playlists:
            playlist = dict(
                pid=p['pid'],
                name=p['name'],
                modified_at=p['modified_at'],
                num_tracks=p['num_tracks'],
                num_albums=p['num_albums'],
                num_edits=p['num_edits'],
                num_artists=p['num_artists'],
                num_followers=p['num_followers'],
                duration_ms=p['duration_ms'],
            )
            track_ids = []
            for t in p['tracks']:
                track_id = t['track_uri']
                track_ids.append(track_id)
                if track_id not in self.tracks_set:
                    self.tracks_set.add(track_id)
                    self.tracks.append(dict(
                        track_id=track_id,
                        track_name=t['track_name'],
                        artist_name=t['artist_name'],
                        lyrics=0,
                    ))
            playlist['track_ids'] = ' '.join(track_ids)
            self.playlists.append(playlist)

    def tokenize(self):
        self.playlists = []
        self.tracks_set = set()
        self.tracks = []
        for index in tqdm(range(self.index_min, self.index_max, self.index_step)):
            index_end = index + 999
            filepath = os.path.join(self.data_dir, 'data', 'mpd.slice.{}-{}.json'.format(index, index_end))
            self.tokenize_file(filepath)

        self.playlists = self.dictifier(self.playlists)
        self.tracks_set = list(self.tracks_set)

        pl_df = pd.DataFrame(self.playlists)
        pl_df.to_csv(os.path.join(self.store_dir, 'playlist.csv'), sep='\t', index=False)

        # tr_set_df = pd.DataFrame(dict(tracks=self.tracks_set))
        # tr_set_df.to_csv(os.path.join(self.store_dir, 'preprocess-tracks.csv'), sep='\t', index=False)

        # tr_df = pd.DataFrame(self.tracks)
        # tr_df.to_csv(os.path.join(self.store_dir, 'tracks-for-lyrics.csv'), sep='\t')

    def get_un_retrieved_tracks(self):
        df = pd.read_csv('data/Spotify/tracks_format.csv', sep='\t')
        retrieved_tracks = set(df.track_id.tolist())

        for index in tqdm(range(self.index_min, self.index_max, self.index_step)):
            index_end = index + 999
            filepath = os.path.join(self.data_dir, 'data', 'mpd.slice.{}-{}.json'.format(index, index_end))
            self.retrieve_tracks_from_file(filepath, retrieved_tracks)

        json.dump(self.un_retrieved_tracks, open('data/Spotify/un_retrieved_tracks.json', 'w'))

    def get_un_retrieved_albums(self):
        df = pd.read_csv('data/Spotify/album.csv', sep='\t')
        retrieved_albums = set(df.album_id.tolist())

        for index in tqdm(range(self.index_min, self.index_max, self.index_step)):
            index_end = index + 999
            filepath = os.path.join(self.data_dir, 'data', 'mpd.slice.{}-{}.json'.format(index, index_end))
            self.retrieve_albums_from_file(filepath, retrieved_albums)

        json.dump(self.un_retrieved_albums, open('data/Spotify/un_retrieved_albums.json', 'w'))
