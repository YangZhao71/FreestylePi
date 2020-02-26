"""
"""
import os

# Credentials
GCLOUD_CREDENTIAL = 'data/gcloud-credential.json'
LUIS_CREDENTIAL = 'data/luis-credential.json'
WEATHER_CREDENTIAL = 'data/weather-credential.json'

def check_file(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError("{} not found".format(filename))


def init_credentials():
    check_file(GCLOUD_CREDENTIAL)
    check_file(LUIS_CREDENTIAL)
    check_file(WEATHER_CREDENTIAL)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCLOUD_CREDENTIAL
    os.environ["LUIS_CREDENTIAL"] = LUIS_CREDENTIAL

    import json
    with open(WEATHER_CREDENTIAL, "r") as file:
        os.environ["WEATHER_APIKEY"] = json.load(file)['api_key']

