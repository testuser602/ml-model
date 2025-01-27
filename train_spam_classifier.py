import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import mlflow
import mlflow.sklearn

# Load dataset
df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])

# Convert labels to binary (spam = 1, ham = 0)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42)

# Convert text to numerical features
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Define a function to train and log the model
def train_and_log_model(C_value, max_iter):
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("C_value", C_value)
        mlflow.log_param("max_iter", max_iter)

        # Train the model
        model = LogisticRegression(C=C_value, max_iter=max_iter, random_state=42)
        model.fit(X_train_vectorized, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log the model
        mlflow.sklearn.log_model(model, "logistic_regression_model")

        # Print results
        print(f"Run with C={C_value}, max_iter={max_iter} -> Accuracy: {accuracy * 100:.2f}%")

# Run experiments with different hyperparameters
hyperparameters = [
    {"C_value": 0.1, "max_iter": 100},
    {"C_value": 1.0, "max_iter": 200},
    {"C_value": 10.0, "max_iter": 300},
]

for params in hyperparameters:
    train_and_log_model(**params)

# Save the vectorizer for future use
with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Vectorizer saved!")
