from flask import Flask, request, jsonify
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
with open("spam_classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data (message) from the POST request
        data = request.json
        message = data.get("message", "")
        if not message:
            return jsonify({"error": "No message provided"}), 400

        # Vectorize the message
        message_vectorized = vectorizer.transform([message])

        # Predict using the model
        prediction = model.predict(message_vectorized)
        label = "spam" if prediction[0] == 1 else "ham"

        # Return the prediction as JSON
        return jsonify({"message": message, "label": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
