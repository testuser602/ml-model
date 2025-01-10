import pickle
import numpy as np


def test_model_prediction():
    # Load the model
    with open("trained_model.pkl", "rb") as file:
        model = pickle.load(file)

    # Test prediction
    new_data = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(new_data)
    assert len(prediction) == 1  # Check if it returns one prediction
