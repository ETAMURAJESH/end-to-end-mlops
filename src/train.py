import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score  


from preprocessing import preprocess_data

#load dataset 

df = pd.read_csv("data/tested.csv")

#preprocess dataset 

df = preprocess_data(df)

#features (input data)
X = df.drop("Survived",axis=1)

# Target (what we predict)
y = df["Survived"]


#splite the dataset in trinning and testing 

X_train, X_test, y_trian ,y_test = train_test_split(
    X,y,test_size=0.2, random_state=42 
)

# creat the model 
model = LogisticRegression(max_iter=1000)

#train model 
model.fit(X_train,y_trian)

#make pridection 
y_pred = model.predict(X_test)

#check accuracy
accuracy = accuracy_score(y_test,y_pred)

print("Model Accuracy:",accuracy)