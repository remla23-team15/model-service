import logging
import pickle

import joblib
from flasgger import Swagger
from flask import Flask, request

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s : %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
swagger = Swagger(app)


@app.route("/", methods=["GET"])
def home():
    """
    Default route
    ---
    consumes:
      - application/json
    responses:
      200:
        description: Some result
    """
    return {
        "result": "Please use the /predict endpoint to predict the sentiment of a review text!",
    }


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict whether a restaurant review is positive or negative.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: review
            properties:
                review:
                    type: string
                    example: This is an example of a restaurant review.
    responses:
      200:
        description: The result of the classification, 'positive' or 'negative'
    """

    # Get data from request
    input_data = request.get_json()
    restaurant_review = input_data.get("review")

    # Load pickle
    log.info("Loading BoW dictionary...")
    cv = pickle.load(open(f"ml_models/c1_BoW_Sentiment_Model.pkl", "rb"))

    # Perform predictions
    log.info("Predictions (via sentiment classifier)...")

    classifier = joblib.load("ml_models/c2_Classifier_Sentiment_Model")
    processed_input = cv.transform([restaurant_review]).toarray()[0]
    prediction = classifier.predict([processed_input])[0]

    # Return result
    prediction_map = {
        0: "negative",
        1: "positive",
    }

    res = {
        "result": prediction_map[prediction],
        "classifier": "GaussianNB",
        "review": restaurant_review
    }

    log.info(res)

    return res


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
