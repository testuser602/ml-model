import pickle
import numpy as np

def test_predict_script():
    # Load the model and vectorizer
    with open("spam_classifier.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    # Test a spam message
    spam_message = "Congratulations! You've won a $1,000 Walmart gift card. Go to http://spam.com to claim now."
    spam_vectorized = vectorizer.transform([spam_message])
    spam_prediction = model.predict(spam_vectorized)
    assert spam_prediction[0] == 1, "Spam message was not classified as spam."

    # Test a ham (non-spam) message
    ham_message = "Hi John, can we meet tomorrow at 10 AM for the project discussion?"
    ham_vectorized = vectorizer.transform([ham_message])
    ham_prediction = model.predict(ham_vectorized)
    assert ham_prediction[0] == 0, "Ham message was incorrectly classified as spam."
