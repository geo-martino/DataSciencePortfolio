# work in progress (WIP)

import json

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

pd.set_option('display.max_columns', None)


def jprint(obj):
    # create a formatted string of the Python JSON object
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)


def auth():
    CLIENT_ID = ''
    CLIENT_SECRET = ''
    AUTH_URL = 'https://accounts.spotify.com/api/token'

    # POST
    auth_response = requests.post(AUTH_URL, {
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
    })

    auth_response_data = auth_response.json()  # convert the response to JSON
    access_token = auth_response_data['access_token']  # return the access token
    return access_token


headers = {
    'Authorization': 'Bearer {token}'.format(token=auth())
}
BASE_URL = 'https://api.spotify.com/v1/'  # base URL of all Spotify API endpoints


def get_audiofeatures(input):
    uri = input[0]
    type = input[1]
    # check for errors in input
    if type not in ('track', 'album', 'artist'):
        print('\33[91;1m' + 'Must of type: artist, album or track' + '\33[0m')
        return

    check_uri = requests.get('https://open.spotify.com/' + type + '/' + uri, headers=headers)
    if check_uri.status_code == 404:
        print('\33[91;1m' + '404: bad URI or incorrect type+URI combo' + '\33[0m')
        return

    feature_list = []
    track_name = []

    def get_features(track_uri):  # print track audio features (key, liveness, danceability, ...)
        f = requests.get(BASE_URL + 'audio-features/' + track_uri, headers=headers).json()
        feature_list.append(pd.DataFrame([f]))

    def get_tracks(album_uri):  # pull all album's tracks
        tracks = requests.get(BASE_URL + 'albums/' + album_uri + '/tracks', headers=headers).json()
        for track in tracks['items']:  # list features of all tracks from this album
            track_name.append(track['name'])
            get_features(track['id'])

    def get_albums(artist_uri):  # pull all artist's albums
        albums = requests.get(BASE_URL + 'artists/' + artist_uri + '/albums', headers=headers,
                              params={'include_groups': 'single', 'limit': 50}).json()
        for album in albums['items']:
            # pull all tracks from this album
            get_tracks(album['id'])

    if type == 'track':
        t = requests.get(BASE_URL + 'tracks/' + uri, headers=headers).json()
        track_name.append(t['album']['name'])
        get_features(uri)
    elif type == 'album':
        get_tracks(uri)
    elif type == 'artist':
        get_albums(uri)

    df = pd.concat(feature_list, ignore_index=True)
    df.insert(0, 'track_name', track_name)

    return df


jorj = ['1odSzdzUpm3ZEEb74GdyiS', 'artist']
ep = ['0rAWaAAMfzHzCbYESj4mfx', 'album']
tws = ['3Xd6KE3ieXLQuxsQx5JRvB', 'track']

features = get_audiofeatures(ep).drop(['type', 'uri', 'track_href', 'analysis_url'], axis=1)
print(features)

plt.figure(figsize=(10, 10))

ax = sns.scatterplot(data=features, x='danceability', y='valence',
                     hue='track_name', palette='rainbow',
                     size='duration_ms', sizes=(50, 1000),
                     alpha=0.7)

# display legend without `size` attribute
h, labs = ax.get_legend_handles_labels()
ax.legend(h[1:5], labs[1:5], loc='best', title='Songs')
plt.show()
