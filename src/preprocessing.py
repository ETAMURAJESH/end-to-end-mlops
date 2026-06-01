import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)


def preprocess_data(df, target_column):

    # SEPARATE FEATURES AND TARGET
    X = df.drop(target_column, axis=1)

    y = df[target_column]


    # DETECT NUMERIC COLUMNS
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns


    # DETECT CATEGORICAL COLUMNS
    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns


    # NUMERIC PIPELINE
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )


    # CATEGORICAL PIPELINE
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )


    # COMBINE PIPELINES
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )


    # TRANSFORM DATA
    X_processed = preprocessor.fit_transform(X)

    # CONVERT SPARSE MATRIX TO DENSE ARRAY
    X_processed = X_processed.toarray()


    return X_processed, y