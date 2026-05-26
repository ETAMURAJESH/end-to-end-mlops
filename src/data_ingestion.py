import pandas as pd 

def load_data(path):
    data = pd.read_csv(path)
    return data 


if __name__== "__main__":

    df = load_data("data/tested.csv")

    print("Dataset Loaded Successfully")

    print(df.head())

    print("/nDataset shape:")
    print(df.shape)

    print("/nMissing values:")
    print(df.isnull().sum())

    print("/ndata Type:")
    print(df.dtypes)
