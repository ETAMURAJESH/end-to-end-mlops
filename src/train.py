import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocessing import preprocess_data


# load data
df = pd.read_csv("data/tested.csv")

# preprocess
df = preprocess_data(df)

# split X and y
X = df.drop("Survived", axis=1)
y = df["Survived"]

# train-test split (FIXED TYPO)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# MLflow tracking
mlflow.start_run()

mlflow.log_param("model", "LogisticRegression")
mlflow.log_param("max_iter", 1000)
mlflow.log_metric("accuracy", accuracy)

# save model
joblib.dump(model, "models/model.pkl")

mlflow.end_run()

print("Model Saved Successfully") 