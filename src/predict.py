import joblib
import pandas as pd 


#load saved model 
model = joblib.load("models/titanic_model.pkl")


#sample passenger data 
sample_data = pd.DataFrame({
    "Pclass": [3],
    "Sex": [1],
    "Age": [22],
    "SibSp": [1],
    "Parch": [0],
    "Fare": [7.25],
    "Embarked": [2]
})

#predict
prediction = model.predict(sample_data)

print("prediction:",prediction)