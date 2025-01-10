import os
import pickle


def test_model_training():
    # Run the training script
    os.system("python train_model.py")

    # Check if the model file was created
    assert os.path.exists("trained_model.pkl")

    # Check if the model can be loaded
    with open("trained_model.pkl", "rb") as file:
        model = pickle.load(file)
        assert model is not None
