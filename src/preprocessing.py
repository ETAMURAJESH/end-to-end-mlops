import pandas as pd

from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):

    # remove unnecessary columns
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    # fill missing Age values
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # fill missing Fare values
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # fill missing Embarked values
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # convert text columns into numbers
    encoder = LabelEncoder()

    df["Sex"] = encoder.fit_transform(df["Sex"])

    df["Embarked"] = encoder.fit_transform(df["Embarked"])

    return df


if __name__ == "__main__":

    df = pd.read_csv("data/tested.csv")

    processed_df = preprocess_data(df)

    print("Preprocessed Dataset:\n")

    print(processed_df.head())

    print("\nMissing Values After Cleaning:")

    print(processed_df.isnull().sum())

    # Save processed dataset
    processed_df.to_csv("data/processed.csv", index=False)

    print("Processed data saved successfully")