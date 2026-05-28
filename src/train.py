import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocessing import preprocess_data

# Load dataset
df = pd.read_csv("data/tested.csv")

# Preprocess dataset
df = preprocess_data(df)

# Features (input data)
X = df.drop("Survived", axis=1)

# Target (what we predict)
y = df["Survived"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Start MLflow tracking
mlflow.start_run()

# Log parameters
mlflow.log_param("max_iter", 1000)

# Log metrics
mlflow.log_metric("accuracy", accuracy)

# Save trained model
joblib.dump(model, "models/model.pkl")

print("Model Saved Successfully")

# End MLflow run
mlflow.end_run()