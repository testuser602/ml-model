import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


def test_train_script():
    # Run the training script
    os.system("python train_spam_classifier.py")
    # Check if the model and vectorizer files are created
    assert os.path.exists("spam_classifier.pkl"), "Model file not created."
    assert os.path.exists("vectorizer.pkl"), "Vectorizer file not created."

    # Verify the saved model
    with open("spam_classifier.pkl", "rb") as model_file:
        model = pickle.load(model_file)
        assert isinstance(model, LogisticRegression), "Saved model is not Logistic Regression."

    # Verify the saved vectorizer
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
        assert isinstance(vectorizer, CountVectorizer), "Saved vectorizer is not CountVectorizer."
