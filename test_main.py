from main import train_model

def test_train_model():
    mse = train_model()
    assert mse > 0, "MSE should be greater than 0"
