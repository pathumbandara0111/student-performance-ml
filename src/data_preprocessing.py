from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "student_performance_features.csv"
)

TARGETS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "student_performance_targets.csv"
)


def load_dataset():
    """Load the raw Student Performance dataset."""

    features = pd.read_csv(FEATURES_PATH)
    targets = pd.read_csv(TARGETS_PATH)

    df = pd.concat([features, targets], axis=1)

    return df


def prepare_target(df):
    """
    Prepare the primary regression target.

    G3 is the final grade.
    G1 and G2 are excluded from the predictor variables
    because they represent previous-period grades.
    """

    target_column = "G3"
    excluded_columns = ["G1", "G2", "G3"]

    X = df.drop(columns=excluded_columns)
    y = df[target_column]

    return X, y


def split_data(X, y, test_size=0.20, random_state=42):
    """Split the dataset into training and testing sets."""

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )


def get_feature_types(X):
    """Identify numerical and categorical features."""

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    return numeric_features, categorical_features


def build_preprocessor(X):
    """
    Build the preprocessing pipeline used by the ML experiments.

    Numerical features:
        StandardScaler

    Categorical features:
        OneHotEncoder
    """

    numeric_features, categorical_features = get_feature_types(X)

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    return preprocessor


def preprocess_data(X_train, X_test, preprocessor):
    """Fit preprocessing on training data and transform both datasets."""

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index
    )

    return X_train_processed, X_test_processed, feature_names


if __name__ == "__main__":

    print("Loading Student Performance dataset...")

    df = load_dataset()

    print(f"Dataset shape: {df.shape}")

    X, y = prepare_target(df)

    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    numeric_features, categorical_features = get_feature_types(X)

    print(f"Numerical features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    preprocessor = build_preprocessor(X)

    X_train_processed, X_test_processed, feature_names = preprocess_data(
        X_train,
        X_test,
        preprocessor
    )

    print(f"Processed training shape: {X_train_processed.shape}")
    print(f"Processed testing shape: {X_test_processed.shape}")
    print(f"Processed feature count: {len(feature_names)}")

    print("\nPreprocessing completed successfully.")
