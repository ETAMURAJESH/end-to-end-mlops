from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV


def get_models(model_names):

    all_models = {

        "LogisticRegression": GridSearchCV(
            LogisticRegression(max_iter=1000),
            {
                "C": [0.1, 1, 10]
            },
            cv=3
        ),

        "DecisionTree": GridSearchCV(
            DecisionTreeClassifier(),
            {
                "max_depth": [3, 5, 10]
            },
            cv=3
        ),

        "RandomForest": GridSearchCV(
            RandomForestClassifier(),
            {
                "n_estimators": [50, 100],
                "max_depth": [5, 10]
            },
            cv=3
        ),

        "KNN": GridSearchCV(
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7]
            },
            cv=3
        ),

        "SVM": GridSearchCV(
            SVC(),
            {
                "kernel": ["linear", "rbf"],
                "C": [0.1, 1, 10]
            },
            cv=3
        ),

        "NaiveBayes": GaussianNB()

    }

    selected_models = {}

    for model_name in model_names:

        selected_models[model_name] = all_models[model_name]

    return selected_models