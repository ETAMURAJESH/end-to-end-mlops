import joblib
import pandas as pd

# LOAD PIPELINE
model = joblib.load("models/pipeline.pkl")

# SAMPLE INPUT DATA
sample = pd.DataFrame([{
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}])

# PREDICT
prediction = model.predict(sample)

print("Prediction:", prediction)
