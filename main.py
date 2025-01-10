from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error


def train_model():
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    return mse


if __name__ == "__main__":
    print(f"Model MSE: {train_model()}")
