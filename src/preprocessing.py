import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def preprocess_data(df, target_column):

    # =========================
    # SPLIT FEATURES / TARGET
    # =========================
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # =========================
    # FIXED FEATURE