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
positive_predictions = 0
negative_predictions = 0
total_predictions = 0
correct_predictions = 0
model_accuracy = 0.0
prediction_duration = 0.0


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

    global positive_predictions, negative_predictions, total_predictions, prediction_duration

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
        positive_predictions += 1
    else:
        negative_predictions += 1

    total_predictions = positive_predictions + negative_predictions

    prediction_duration = round(time.time() - start_time, 4)

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
    global total_predictions, correct_predictions, model_accuracy

    # Get data from request
    input_data = request.get_json()
    prediction_feedback = input_data.get("feedback")

    # Update the metrics
    log.info("Update model accuracy with given feedback...")

    if prediction_feedback:
        correct_predictions += 1

    # Avoid division by 0
    if total_predictions == 0:
        return Response("You must perform a prediction first before giving feedback!", status=400)

    model_accuracy = round(correct_predictions / total_predictions, 2)

    log.info(f"Model accuracy = {model_accuracy}")

    return {
        "model_accuracy": model_accuracy
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

    global positive_predictions, negative_predictions, total_predictions, \
        model_accuracy, prediction_duration, correct_predictions

    prometheus_metrics = "# HELP positive_predictions Total positive predictions.\n"
    prometheus_metrics += "# TYPE positive_predictions counter\n"
    prometheus_metrics += f"positive_predictions {positive_predictions}\n\n"

    prometheus_metrics += "# HELP negative_predictions Total negative predictions.\n"
    prometheus_metrics += "# TYPE negative_predictions counter\n"
    prometheus_metrics += f"negative_predictions {negative_predictions}\n\n"

    prometheus_metrics += "# HELP total_predictions Total predictions.\n"
    prometheus_metrics += "# TYPE total_predictions counter\n"
    prometheus_metrics += f"total_predictions {total_predictions}\n\n"

    prometheus_metrics += "# HELP correct_predictions Total correct predictions based on feedback.\n"
    prometheus_metrics += "# TYPE correct_predictions counter\n"
    prometheus_metrics += f"correct_predictions {correct_predictions}\n\n"

    prometheus_metrics += "# HELP model_accuracy The predictions accuracy.\n"
    prometheus_metrics += "# TYPE model_accuracy gauge\n"
    prometheus_metrics += f"model_accuracy {model_accuracy}\n\n"

    prometheus_metrics += "# HELP prediction_duration The predictions durtion in seconds.\n"
    prometheus_metrics += "# TYPE prediction_duration histogram\n"
    prometheus_metrics += f"prediction_duration {prediction_duration}\n"

    return Response(prometheus_metrics, mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
