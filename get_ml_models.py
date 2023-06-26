""" get_ml_models.py """
import os

import nextcloud_client
from version_util_python.version_util import VersionUtil


def get_ml_models():
    """
    Download ML models based on latest version from cloud repository.
    """
    # Create ml_models directory if does not exists
    if not os.path.exists("./ml_models"):
        os.makedirs("./ml_models")

    # Initialize version util class
    versions_lib = VersionUtil()
    print(f"Downloading ML models version {versions_lib.model_training_version}...")

    try:
        # Initialize remote repository client
        repository_url = "https://liv.nl.tab.digital/s/TyPqR5HCjExqNQq"
        nc_client = nextcloud_client.Client.from_public_link(repository_url)

        # Download models
        nc_client.get_file(
            f"/{versions_lib.model_training_version}/c1_BoW_Sentiment_Model.pkl",
            "./ml_models/c1_BoW_Sentiment_Model.pkl"
        )
        nc_client.get_file(
            f"/{versions_lib.model_training_version}/c2_Classifier_Sentiment_Model",
            "./ml_models/c2_Classifier_Sentiment_Model"
        )

        print("Done, the ml_models folder and the ML models were downloaded successfully!")
    except nextcloud_client.nextcloud_client.HTTPResponseError:
        print("An error occurred while downloading the models, please try again later.")
        return False, "An error occurred while downloading the models, please try again later."

    return True, f"Models updated to version {versions_lib.model_training_version}"


if __name__ == '__main__':
    get_ml_models()
