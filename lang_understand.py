import json
from pprint import pprint
from os.path import isfile

from azure.cognitiveservices.language.luis.runtime import LUISRuntimeClient
from msrest.authentication import CognitiveServicesCredentials


class LangUnderstand:
    def __init__(self):
        import os
        credential = os.getenv('LUIS_CREDENTIAL')
        with open(credential, "r") as file:
            self.credential = json.load(file)

        self.client = LUISRuntimeClient(
            self.credential['endpoint_url'],
            CognitiveServicesCredentials(self.credential['subscription_key'])
        )

    def predict(self, query=""):
        result = self.client.prediction.resolve(
            self.credential['application_id'],  # LUIS Application ID
            query,
            staging=True
        )

        return result

    def predict_example(self, query=""):
        try:
            print("Executing query: {}".format(query))
            result = self.client.prediction.resolve(
                self.credential['application_id'],  # LUIS Application ID
                query,
                staging=True
            )

            print("\nDetected intent: {} (score: {:d}%)".format(
                result.top_scoring_intent.intent,
                int(result.top_scoring_intent.score*100)
            ))
            print("Detected entities:")
            for entity in result.entities:
                print("\t-> Entity '{}' (type: {}, score:{:d}%)".format(
                    entity.entity,
                    entity.type,
                    int(entity.additional_properties['score']*100)
                ))
            print("\nComplete result object as dictionnary")
            pprint(result.as_dict())

        except Exception as err:
            print("Encountered exception. {}".format(err))


if __name__ == "__main__":
    from utils.credentials import init_credentials
    init_credentials()

    luis = LangUnderstand()
    luis.predict_example('can you rap about weather')
