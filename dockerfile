# Use an official Python runtime as a parent image
FROM python:3.9.6

# Set the working directory
WORKDIR /app

# Copy the requirements and application code to the container
COPY requirements.txt .
COPY app.py .
COPY spam_classifier.pkl .
COPY vectorizer.pkl .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
