import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from preprocessing import preprocess_data
from model_factory import get_models
from config_loader import load_config


# LOAD CONFIG
config = load_config()


# LOAD DATASET
dataset_path = config["dataset"]["path"]

df = pd.read_csv(dataset_path)


# TARGET COLUMN
target_column = config["target_column"]


# PREPROCESS DATA
X, y = preprocess_data(df, target_column)


# TRAIN TEST SPLIT
test_size = config["train"]["test_size"]

random_state = config["train"]["random_state"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)


# LOAD MODELS FROM CONFIG
model_names = config["models"]

models = get_models(model_names)


# BEST MODEL TRACKING
best_model = None

best_accuracy = 0

best_model_name = ""


# TRAIN ALL MODELS
for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"{model_name} Accuracy: {accuracy}")


    # SHOW BEST PARAMETERS
    if hasattr(model, "best_params_"):

        print(f"{model_name} Best Params: {model.best_params_}")


    # MLFLOW LOGGING
    mlflow.start_run(run_name=model_name)

    mlflow.log_param("model_name", model_name)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.end_run()


    # BEST MODEL CHECK
    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = model_name


# FINAL RESULTS
print("\n==========================")

print(f"BEST MODEL: {best_model_name}")

print(f"BEST ACCURACY: {best_accuracy}")

print("==========================")


# SAVE BEST MODEL
joblib.dump(best_model, "models/model.pkl")

print("Best model saved successfully")