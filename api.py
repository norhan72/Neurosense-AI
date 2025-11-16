from flask import Flask, request, jsonify
from pyngrok import ngrok
from survey_questions import model

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data["features"]
    prediction = model.predict([features]).tolist()
    return jsonify({"prediction": prediction})


public_url = ngrok.connect(5000)
print("Public URL:", public_url)

app.run(port=5000)
