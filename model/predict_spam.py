import pickle

# Load the model and vectorizer
with open("spam_classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


# Function to predict spam
def predict_spam(message):
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)
    return "Spam" if prediction[0] == 1 else "Not Spam"


# Example usage
new_message = "Once in a blue moon hi"
print(f"Message: {new_message}")
print(f"Prediction: {predict_spam(new_message)}")
