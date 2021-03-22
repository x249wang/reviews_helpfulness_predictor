import json
from flask import Flask, Response, request, jsonify, render_template
import utils

PORT = 8080

app = Flask(__name__)


@app.route("/")
def root():
    return {"greetings": "Welcome to the app!"}


# Reference: https://flask.palletsprojects.com/en/1.1.x/patterns/apierrors/
class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv["message"] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/docs")
def get_docs():
    return render_template("swaggerui.html")


@app.route("/predict", methods=["POST"])
def api():
    input_data = request.json

    if ("summary" not in input_data) or ("review" not in input_data):
        raise InvalidUsage(
            "Must specify summary and review in the request", status_code=422
        )

    sanitized_text = utils.clean_text(input_data["summary"], input_data["review"])

    if sanitized_text:
        language = utils.detect_language(sanitized_text)
        prediction = utils.generate_prediction_from_text(sanitized_text)
        return Response(
            json.dumps(
                {
                    "prediction": prediction,
                    "language_detected": language,
                    "disclaimer": "Prediction not accurate for non-English reviews",
                }
            )
        )

    else:
        raise InvalidUsage(
            "Review content cannot be blank (must provide either the summary or text section)",
            status_code=422,
        )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=PORT)
