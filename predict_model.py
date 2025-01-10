import pickle
import numpy as np

# Load the trained model
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# Example input for prediction (a single data point from the Iris dataset)
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])

# Make predictions
predicted_class = model.predict(new_data)
print(f"Predicted Class: {predicted_class[0]}")
