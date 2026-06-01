import joblib
import pandas as pd

# Load best model
pipeline = joblib.load("models/pipeline.pkl")

sample_data = pd.DataFrame({
    "Pclass": [3],
    "Age": [22],
    "SibSp": [1],
    "Parch": [0],
    "Fare": [7.25],
    "Sex": ["male"],
    "Embarked": ["S"]
})

print("Input Data:")
print(sample_data)

prediction = model.predict(sample_data)

print("Prediction:", prediction)