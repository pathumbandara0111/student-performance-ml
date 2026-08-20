import pandas as pd


def create_at_risk_target(
    df: pd.DataFrame,
    target_column: str = "G3",
    threshold: int = 10
) -> pd.Series:
    """
    Create the binary At-Risk classification target.

    Students with a final grade below the threshold are classified
    as At Risk.

    G3 < 10  -> 1 (At Risk)
    G3 >= 10 -> 0 (Not At Risk)
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset."
        )

    return (df[target_column] < threshold).astype(int)


def prepare_classification_data(
    df: pd.DataFrame,
    target_column: str = "G3",
    threshold: int = 10
):
    """
    Prepare features and target for the classification task.

    G1 and G2 are excluded because they represent previous-period
    academic grades and are not part of the primary early-prediction
    experiment.
    """

    excluded_columns = ["G1", "G2", target_column]

    missing_columns = [
        column
        for column in excluded_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Expected columns not found: {missing_columns}"
        )

    X = df.drop(columns=excluded_columns)

    y = create_at_risk_target(
        df,
        target_column=target_column,
        threshold=threshold
    )

    return X, y


def get_class_distribution(y: pd.Series) -> pd.Series:
    """Return the class distribution."""

    return y.value_counts().sort_index()


def get_class_distribution_percent(y: pd.Series) -> pd.Series:
    """Return the class distribution as percentages."""

    return (
        y.value_counts(normalize=True)
        .sort_index()
        .mul(100)
    )


if __name__ == "__main__":

    from data_preprocessing import load_dataset

    print("Loading dataset...")

    df = load_dataset()

    X, y = prepare_classification_data(df)

    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    print("\nClass distribution:")
    print(get_class_distribution(y))

    print("\nClass distribution (%):")
    print(get_class_distribution_percent(y).round(2))

    print("\nClassification feature columns:")
    print(list(X.columns))

    print("\nFeature engineering completed successfully.")
