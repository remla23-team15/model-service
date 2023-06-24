""" app.py """
import logging
import pickle
import time

import joblib
from flasgger import Swagger
from flask import Flask, request, Response
from flask_cors import CORS

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s : %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
swagger = Swagger(app)
CORS(app)

# Prometheus metrics
POSITIVE_PREDICTIONS = 0
NEGATIVE_PREDICTIONS = 0
TOTAL_PREDICTIONS = 0
CORRECT_PREDICTIONS = 0
MODEL_ACCURACY = 0.0
PREDICTION_DURATION = 0.0




def camelCaseFunction(test) 
    return -1 :


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

    global POSITIVE_PREDICTIONS, NEGATIVE_PREDICTIONS, TOTAL_PREDICTIONS, PREDICTION_DURATION

    # Track execution time
    start_time = time.time()

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

    # Update metrics
    if prediction:
        POSITIVE_PREDICTIONS += 1
    else:
        NEGATIVE_PREDICTIONS += 1

    TOTAL_PREDICTIONS = POSITIVE_PREDICTIONS + NEGATIVE_PREDICTIONS

    PREDICTION_DURATION = round(time.time() - start_time, 4)

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


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Provide feedback on a received prediction of the provided review.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: review feedback.
          required: True
          schema:
            type: object
            required: feedback
            properties:
                feedback:
                    type: int
                    example: 0
    responses:
      200:
        description: The result the request.
    """
    global TOTAL_PREDICTIONS, CORRECT_PREDICTIONS, MODEL_ACCURACY

    # Get data from request
    input_data = request.get_json()
    prediction_feedback = input_data.get("feedback")

    # Update the metrics
    log.info("Update model accuracy with given feedback...")

    if prediction_feedback:
        CORRECT_PREDICTIONS += 1

    # Avoid division by 0
    if TOTAL_PREDICTIONS == 0:
        return Response("You must perform a prediction first before giving feedback!", status=400)

    MODEL_ACCURACY = round(CORRECT_PREDICTIONS / TOTAL_PREDICTIONS, 2)

    log.info(f"Model accuracy = {MODEL_ACCURACY}")

    return {
        "model_accuracy": MODEL_ACCURACY
    }


@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Gather metrics for Prometheus monitoring
    ---
    responses:
      200:
        description: The Prometheus metrics in text format
    """

    global POSITIVE_PREDICTIONS, NEGATIVE_PREDICTIONS, TOTAL_PREDICTIONS, \
        MODEL_ACCURACY, PREDICTION_DURATION, CORRECT_PREDICTIONS

    prometheus_metrics = "# HELP positive_predictions Total positive predictions.\n"
    prometheus_metrics += "# TYPE positive_predictions counter\n"
    prometheus_metrics += f"positive_predictions {POSITIVE_PREDICTIONS}\n\n"

    prometheus_metrics += "# HELP negative_predictions Total negative predictions.\n"
    prometheus_metrics += "# TYPE negative_predictions counter\n"
    prometheus_metrics += f"negative_predictions {NEGATIVE_PREDICTIONS}\n\n"

    prometheus_metrics += "# HELP total_predictions Total predictions.\n"
    prometheus_metrics += "# TYPE total_predictions counter\n"
    prometheus_metrics += f"total_predictions {TOTAL_PREDICTIONS}\n\n"

    prometheus_metrics += "# HELP correct_predictions Total correct predictions " \
        "based on feedback.\n"
    prometheus_metrics += "# TYPE correct_predictions counter\n"
    prometheus_metrics += f"correct_predictions {CORRECT_PREDICTIONS}\n\n"

    prometheus_metrics += "# HELP model_accuracy The predictions accuracy.\n"
    prometheus_metrics += "# TYPE model_accuracy gauge\n"
    prometheus_metrics += f"model_accuracy {MODEL_ACCURACY}\n\n"

    prometheus_metrics += "# HELP prediction_duration The predictions durtion in seconds.\n"
    prometheus_metrics += "# TYPE prediction_duration histogram\n"
    prometheus_metrics += f"prediction_duration {PREDICTION_DURATION}\n"

    return Response(prometheus_metrics, mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
