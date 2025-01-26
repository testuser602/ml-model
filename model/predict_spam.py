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
new_message = "Congratulations! You've won a $1,000 Walmart gift card. Go to http://spam.com to claim now."
print(f"Message: {new_message}")
print(f"Prediction: {predict_spam(new_message)}")
