import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):

    # drop unnecessary columns
    df = df.drop(
        ["PassengerId", "Name", "Ticket", "Cabin"],
        axis=1,
        errors="ignore"
    )

    # fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # encode categorical columns
    encoder = LabelEncoder()

    df["Sex"] = encoder.fit_transform(df["Sex"])
    df["Embarked"] = encoder.fit_transform(df["Embarked"])

    return df


# test run (only when file executed directly)
if __name__ == "__main__":

    df = pd.read_csv("data/tested.csv")
    processed_df = preprocess_data(df)

    print(processed_df.head())
    print(processed_df.isnull().sum())

    processed_df.to_csv("data/processed.csv", index=False)
    print("Saved: data/processed.csv")