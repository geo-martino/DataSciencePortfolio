import json
import time

import pandas as pd
import requests
from IPython.core.display import clear_output
from tqdm import tqdm

pd.set_option('display.max_columns', None)


def lastfm_get(payload):
    API_KEY = ''
    USER_AGENT = 'jor-mar-mar'

    # define headers and URL
    headers = {'user-agent': USER_AGENT}
    url = 'http://ws.audioscrobbler.com/2.0/'

    # Add API key and format to the payload
    payload['api_key'] = API_KEY
    payload['format'] = 'json'

    response = requests.get(url, headers=headers, params=payload)
    return response


def jprint(obj):
    # create a formatted string of the Python JSON object
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)


def get_artists(total_pages):
    responses = []
    page = 1

    while page <= total_pages:
        payload = {
            'method': 'chart.gettopartists',
            'limit': 500,
            'page': page
        }

        print("Requesting page {}/{}".format(page, total_pages))  # print some output so we can see the status
        clear_output(wait=True)  # clear the output to make things neater

        response = lastfm_get(payload)  # make the API call

        if response.status_code != 200:  # if we get an error, print the response and halt the loop
            print(response.text)
            break

        page = int(response.json()['artists']['@attr']['page'])  # extract pagination info
        # total_pages = int(response.json()['artists']['@attr']['totalPages'])

        responses.append(response)  # append response

        if getattr(response, 'from_cache', True):  # if it's not a cached result, sleep
            time.sleep(0.25)

        page += 1  # increment the page number

        frames = [pd.DataFrame(r.json()['artists']['artist']) for r in responses]

    artists = pd.concat(frames).drop('image', axis=1).drop_duplicates().reset_index(drop=True)
    return artists


def lookup_tags(artist):
    response = lastfm_get({
        'method': 'artist.getTopTags',
        'artist': artist
    })

    # if there's an error, return error code
    if response.status_code != 200:
        error = "error ", response.status_code
        return error

    # extract the top three tags and turn them into a string
    tags = [t['name'] for t in response.json()['toptags']['tag'][:3]]
    tags_str = ', '.join(tags)

    # rate limiting
    if not getattr(response, 'from_cache', False):
        time.sleep(0.25)
    return tags_str


artists = get_artists(3)
tqdm.pandas()
artists['tags'] = artists['name'].progress_apply(lookup_tags)  # look up top 3 tags for data pulled with get_artists
artists[["playcount", "listeners"]] = artists[["playcount", "listeners"]].astype(int)  # convert data type to int
artists = artists.sort_values("listeners", ascending=False)  # sort data by most listeners
artists.to_csv('output/artists.csv', index=False)  # export csv
