import requests, json
from datetime import datetime

astros = requests.get("http://api.open-notify.org/astros.json")

# get status code of API
print(astros.status_code)

def jprint(obj):
    # create a formatted string of the Python JSON object
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)

jprint(astros.json())

# next API takes lat and lon coords as input
# define London coords
parameters = {
    "lat": 51.5074,
    "lon": 0.1278
}

coords = requests.get("http://api.open-notify.org/iss-pass.json", params=parameters)

# return only the response
pass_times = coords.json()
jprint(pass_times)

# return only the risetimes formatted into YYYY-MM-DD HH:MM:SS using datetime
risetimes = []
"""
for d in pass_times:
    rt = d['risetime']
    time = datetime.fromtimestamp(rt)
    risetimes.append(time)
    print(time)
"""
for d in pass_times['response']:
    print(d['duration'])
