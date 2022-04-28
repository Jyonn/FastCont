import datetime
import json
import multiprocessing
import os.path
import time
from typing import Union

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

AUTHS = []
with open('auth.txt', 'r') as f:
    for line in f:
        if line[-1] == '\n':
            line = line[:-1]
        AUTHS.append(json.loads(line))

INDEX = 0
TOTAL = len(AUTHS)


class Auth:
    def __init__(self, index: int):
        self.index = index
        self.access_token = None
        self.expire_time = 0
        self.sleep = 0.8


class ArtistFetcher:
    def __init__(self):
        self.cols = ['artist_id', 'num_followers', 'genres', 'name', 'popularity']
        self.base_dir = 'data/Spotify'
        self.track_path = os.path.join(self.base_dir, 'tracks_format.csv')
        self.artist_path = os.path.join(self.base_dir, 'artist.csv')

        self.tracks = pd.read_csv(self.track_path, sep='\t')

        self.artists = set()
        for artist_ids in self.tracks.artist_ids.tolist():
            artist_ids = set(artist_ids.split(' '))
            self.artists.update(artist_ids)

        self.fetched_artists = self.load_fetched_artists()
        # self.fetched_artists = []
        print('Total', len(self.artists))
        print('Fetched', len(self.fetched_artists))

        self.num_workers = 1
        self.auths = [Auth(index=(i + INDEX) % TOTAL) for i in range(self.num_workers)]

    def load_fetched_artists(self):
        return set(pd.read_csv(self.artist_path, sep='\t').artist_id.tolist())

    def switch_next(self, i_worker):
        auth = self.auths[i_worker]
        with open('switch.log', 'a') as f:
            f.write('No. {}, SWITCH FROM {} TO '.format(i_worker, auth.index))
            auth.index = (auth.index + self.num_workers) % TOTAL
            f.write('{}\n'.format(auth.index))
        self.refresh_access_token(i_worker)

    @staticmethod
    def get_time():
        return datetime.datetime.now().timestamp()

    def refresh_access_token(self, i_worker: int):
        auth = self.auths[i_worker]
        url = 'https://accounts.spotify.com/api/token'
        r = requests.post(
            url=url,
            headers=dict(Authorization='Basic ' + AUTHS[auth.index]['auth']),
            data=dict(
                grant_type='refresh_token',
                refresh_token=AUTHS[auth.index]['refresh'],
            )
        )
        data = r.json()
        r.close()

        auth.expire_time = self.get_time() + data['expires_in'] - 60
        auth.access_token = data['token_type'] + ' ' + data['access_token']

    def _fetch_artist(self, artist_id, i_worker) -> Union[str, dict]:
        auth = self.auths[i_worker]
        url = 'https://api.spotify.com/v1/artists/' + artist_id
        r = requests.get(
            url=url,
            headers=dict(Authorization=auth.access_token)
        )
        if r.content.decode() == 'Too many requests':
            data = 'Too many requests'
        else:
            data = r.json()
        r.close()
        return data

    def fetch_artist(self, artist_id, i_worker):
        auth = self.auths[i_worker]
        if self.get_time() > auth.expire_time:
            self.refresh_access_token(i_worker)

        data = None
        for _ in range(3):
            try:
                data = self._fetch_artist(artist_id, i_worker)
                if data == 'Too many requests':
                    self.switch_next(i_worker)
                else:
                    break
            except Exception:
                pass

        if not data or (isinstance(data, str)):
            return None

        if 'error' in data:
            if 'rate' in data['error']['message']:
                auth.sleep += 0.05
            print(data)
            return None

        return dict(
            artist_id=artist_id,
            name=data['name'],
            genres='@'.join(data['genres']),
            popularity=data['popularity'],
            num_followers=data['followers']['total'],
        )

    def save_artist(self, data, i_worker):
        with open(self.artist_path + str(i_worker), 'a+') as f:
            s = []
            for col in self.cols:
                data[col] = str(data[col])
                if '\t' in data[col]:
                    data[col] = data[col].replace('\t', ' ')
                    print(data['artist_id'], 'meets warning \\t')
                s.append(data[col])
            s = '\t'.join(s) + '\n'
            f.write(s)

    def worker(self, artists, i_worker):
        for artist_id in tqdm(artists):
            time.sleep(self.auths[i_worker].sleep)
            artist_data = self.fetch_artist(artist_id, i_worker)
            if artist_data:
                self.save_artist(artist_data, i_worker)

    def fetch(self):
        working_artists = []
        for artist_id in self.artists:
            if artist_id not in self.fetched_artists:
                working_artists.append(artist_id)
        indexes = np.arange(self.num_workers) * (len(working_artists) // self.num_workers)
        indexes = indexes.tolist()
        indexes.append(len(working_artists))
        print(indexes)

        for i in range(self.num_workers):
            processor = multiprocessing.Process(target=self.worker, args=(working_artists[indexes[i]: indexes[i+1]], i))
            processor.start()


class AlbumFetcher:
    def __init__(self):
        self.cols = ['album_id', 'label', 'name', 'popularity', 'release_date']
        self.base_dir = 'data/Spotify'
        self.track_path = os.path.join(self.base_dir, 'tracks_format.csv')
        self.album_path = os.path.join(self.base_dir, 'album.csv')

        self.tracks = pd.read_csv(self.track_path, sep='\t')
        self.albums = list(set(self.tracks.album_id.tolist()))

        self.fetched_albums = self.load_fetched_albums()
        print('Total', len(self.albums))
        print('Fetched', len(self.fetched_albums))

        self.num_workers = 1
        self.auths = [Auth(index=(i + INDEX) % TOTAL) for i in range(self.num_workers)]

        self.un_retrieved_albums = json.load(open('data/Spotify/un_retrieved_albums.json'))

    def load_fetched_albums(self):
        return set(pd.read_csv(self.album_path, sep='\t').album_id.tolist())

    def switch_next(self, i_worker):
        auth = self.auths[i_worker]
        with open('switch.log', 'w') as f:
            f.write('No. {}, SWITCH FROM {} TO '.format(i_worker, auth.index))
            auth.index = (auth.index + self.num_workers) % TOTAL
            f.write('{}\n'.format(auth.index))
        self.refresh_access_token(i_worker)

    @staticmethod
    def get_time():
        return datetime.datetime.now().timestamp()

    def refresh_access_token(self, i_worker: int):
        auth = self.auths[i_worker]
        url = 'https://accounts.spotify.com/api/token'
        r = requests.post(
            url=url,
            headers=dict(Authorization='Basic ' + AUTHS[auth.index]['auth']),
            data=dict(
                grant_type='refresh_token',
                refresh_token=AUTHS[auth.index]['refresh'],
            )
        )
        data = r.json()
        r.close()

        auth.expire_time = self.get_time() + data['expires_in'] - 60
        auth.access_token = data['token_type'] + ' ' + data['access_token']

    def _fetch_album(self, album_id, i_worker) -> Union[str, dict]:
        auth = self.auths[i_worker]
        url = 'https://api.spotify.com/v1/albums/' + album_id
        r = requests.get(
            url=url,
            headers=dict(Authorization=auth.access_token)
        )
        if r.content.decode() == 'Too many requests':
            data = 'Too many requests'
        else:
            data = r.json()
        r.close()
        return data

    def fetch_album(self, album_id, i_worker):
        auth = self.auths[i_worker]
        if self.get_time() > auth.expire_time:
            self.refresh_access_token(i_worker)

        data = None
        for _ in range(3):
            try:
                data = self._fetch_album(album_id, i_worker)
                if data == 'Too many requests':
                    self.switch_next(i_worker)
                else:
                    break
            except Exception:
                pass

        if not data or (isinstance(data, str)):
            return None

        if 'error' in data:
            if data['error']['message'] == 'non existing id':
                assert album_id in self.un_retrieved_albums
                print('get from un retrieved albums')
                data = self.un_retrieved_albums[album_id]
                return dict(
                    album_id=album_id,
                    name=data['name'],
                    label='',
                    popularity='',
                    release_date=''
                )
            print('other errors for [{}]'.format(album_id), data)
            return None

        return dict(
            album_id=album_id,
            name=data['name'],
            label=data['label'],
            popularity=data['popularity'],
            release_date=data['release_date']
        )

    def save_album(self, data, i_worker):
        with open(self.album_path + str(i_worker), 'a+') as f:
            s = []
            for col in self.cols:
                data[col] = str(data[col])
                if '\t' in data[col]:
                    data[col] = data[col].replace('\t', ' ')
                    print(data['album_id'], 'meets warning \\t')
                s.append(data[col])
            s = '\t'.join(s) + '\n'
            f.write(s)

    def worker(self, albums, i_worker):
        for album_id in tqdm(albums):
            time.sleep(self.auths[i_worker].sleep)
            album_data = self.fetch_album(album_id, i_worker)
            if album_data:
                self.save_album(album_data, i_worker)

    def fetch(self):
        working_albums = []
        for album_id in self.albums:
            if album_id not in self.fetched_albums:
                working_albums.append(album_id)
        indexes = np.arange(self.num_workers) * (len(working_albums) // self.num_workers)
        indexes = indexes.tolist()
        indexes.append(len(working_albums))
        print(indexes)

        for i in range(self.num_workers):
            processor = multiprocessing.Process(target=self.worker, args=(working_albums[indexes[i]: indexes[i+1]], i))
            processor.start()


class TrackFetcher:
    def __init__(self, data_dir, store_dir):
        self.cols = ['track_id', 'album_id', 'artist_ids', 'duration_ms', 'name', 'popularity']
        self.data_dir = data_dir
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)

        self.access_token = None
        self.expire_time = 0

        self.track_preview_dir = os.path.join(self.store_dir, 'track-preview')
        os.makedirs(self.track_preview_dir, exist_ok=True)

        self.track_data_path = os.path.join(self.store_dir, 'tracks_format.csv')

        self.fetched_tracks = self.load_fetched_tracks()
        self.un_retrieved_tracks = json.load(open('data/Spotify/un_retrieved_tracks.json'))
        print('already:', len(self.fetched_tracks))

    def switch_next(self):
        global INDEX
        INDEX = (INDEX + 1) % TOTAL
        print('SWITCH TO', INDEX)
        with open('switch.log', 'a') as f:
            f.write('switch to %s\n' % INDEX)
        self.refresh_access_token()

    def load_fetched_tracks(self):
        df = pd.read_csv(self.track_data_path, sep='\t')
        return set(df.track_id.tolist())

    @staticmethod
    def get_time():
        return datetime.datetime.now().timestamp()

    def refresh_access_token(self):
        url = 'https://accounts.spotify.com/api/token'
        r = requests.post(
            url=url,
            headers=dict(Authorization='Basic ' + AUTHS[INDEX]['auth']),
            data=dict(
                grant_type='refresh_token',
                refresh_token=AUTHS[INDEX]['refresh'],
            )
        )
        data = r.json()
        r.close()

        self.expire_time = self.get_time() + data['expires_in'] - 60
        self.access_token = data['token_type'] + ' ' + data['access_token']

    def _fetch_preview(self, track_id, preview_url):
        with requests.get(preview_url) as r:
            with open(os.path.join(self.track_preview_dir, track_id), 'wb+') as f:
                f.write(r.content)
                f.flush()

    def _fetch_track(self, track_id) -> Union[str, dict]:
        url = 'https://api.spotify.com/v1/tracks/' + track_id
        r = requests.get(
            url=url,
            headers=dict(Authorization=self.access_token)
        )
        print(r.json())
        if r.content.decode() == 'Too many requests':
            data = 'Too many requests'
        else:
            data = r.json()
        r.close()
        return data

    def fetch_track(self, track_id):
        # if track_id in self.fetched_tracks:
        #     return
        #
        if self.get_time() > self.expire_time:
            self.refresh_access_token()

        data = None
        for _ in range(3):
            try:
                data = self._fetch_track(track_id)
                if data == 'Too many requests':
                    self.switch_next()
                else:
                    break
            except Exception:
                pass

        if not data or (isinstance(data, str)):
            return None

        if 'error' in data and data['error'].get('message') == 'non existing id':
            assert track_id in self.un_retrieved_tracks
            print('get from un retrieved tracks')
            data = self.un_retrieved_tracks[track_id]
            return dict(
                track_id=track_id,
                album_id=data['album_id'],
                artist_ids=data['artist_ids'],
                duration_ms='',
                name=data['name'],
                popularity=''
            )
        elif 'error' in data:
            print('other errors for', track_id)
            return None

        if data.get('preview_url'):
            try:
                self._fetch_preview(track_id, data['preview_url'])
            except Exception:
                pass

        return dict(
            track_id=track_id,
            album_id=data['album']['uri'][len('spotify:album:'):],
            artist_ids=' '.join([artist['uri'][len('spotify:artist:'):] for artist in data['artists']]),
            duration_ms=data['duration_ms'],
            name=data['name'],
            popularity=data['popularity'],
        )

    def save_track(self, data, i_worker):
        with open(self.track_data_path + str(i_worker), 'a+') as f:
            s = []
            for col in self.cols:
                data[col] = str(data[col])
                if '\t' in data[col]:
                    data[col] = data[col].replace('\t', ' ')
                    print(data['track_id'], 'meets warning \\t')
                s.append(data[col])
            s = '\t'.join(s) + '\n'
            f.write(s)

    def worker(self, tracks, i_worker):
        for track_id in tqdm(tracks):
            time.sleep(1)
            track_data = self.fetch_track(track_id)
            if track_data:
                self.save_track(track_data, i_worker)

    def fetch(self, num_workers=5):
        track_path = os.path.join(self.data_dir, 'preprocess-tracks.csv')
        with open(track_path, 'r') as f:
            tracks = f.read().split('\n')[1:]

        working_tracks = []
        for track in tracks:
            track_id = track[len('spotify:track:'):]
            if track_id not in self.fetched_tracks:
                working_tracks.append(track_id)
        indexes = np.arange(num_workers) * (len(working_tracks) // num_workers)
        indexes = indexes.tolist()
        indexes.append(len(working_tracks))
        print(indexes)

        for i in range(num_workers):
            processor = multiprocessing.Process(target=self.worker, args=(working_tracks[indexes[i]: indexes[i+1]], i))
            processor.start()

        # for track in tqdm(tracks):
        #     track_id = track[len('spotify:track:'):]
        #     track_data = self.fetch_track(track_id)
        #     if track_data:
        #         self.save_track(track_data)
