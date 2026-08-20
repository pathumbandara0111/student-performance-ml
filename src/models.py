from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def get_regression_models(random_state=42):
    """
    Return the regression models used in the project.

    Returns
    -------
    dict
        Dictionary containing the candidate regression models.
    """

    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=random_state
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=random_state
        ),
    }

    return models


def get_classification_models(random_state=42):
    """
    Return the classification models used in the project.

    Returns
    -------
    dict
        Dictionary containing the candidate classification models.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        ),
    }

    return models


def get_tuned_regression_model(random_state=42):
    """
    Return the selected tuned Random Forest regression model.

    Hyperparameters correspond to the final regression experiment.
    """

    return RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=4,
        max_features=0.5,
        random_state=random_state
    )


def get_tuned_classification_model(random_state=42):
    """
    Return the selected tuned Random Forest classification model.

    Hyperparameters correspond to the final classification experiment.
    """

    return RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features="log2",
        random_state=random_state
    )


def get_all_models(random_state=42):
    """
    Return all candidate and tuned models used in the project.
    """

    models = {}

    models["regression"] = get_regression_models(
        random_state=random_state
    )

    models["classification"] = get_classification_models(
        random_state=random_state
    )

    models["tuned_regression"] = get_tuned_regression_model(
        random_state=random_state
    )

    models["tuned_classification"] = get_tuned_classification_model(
        random_state=random_state
    )

    return models


if __name__ == "__main__":

    print("Available regression models:")

    for name in get_regression_models():
        print(f"- {name}")

    print("\nAvailable classification models:")

    for name in get_classification_models():
        print(f"- {name}")

    print("\nTuned regression model:")
    print(get_tuned_regression_model())

    print("\nTuned classification model:")
    print(get_tuned_classification_model())

    print("\nModel definitions loaded successfully.")
