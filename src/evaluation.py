import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score


def regression_metrics(y_true, y_pred):
    """
    Calculate regression evaluation metrics.

    Returns
    -------
    dict
        MAE, RMSE and R².
    """

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def classification_metrics(
    y_true,
    y_pred,
    at_risk_label=1
):
    """
    Calculate classification evaluation metrics.

    Label convention:
        0 = Not At Risk
        1 = At Risk
    """

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    at_risk_precision = precision_score(
        y_true,
        y_pred,
        pos_label=at_risk_label,
        zero_division=0
    )

    at_risk_recall = recall_score(
        y_true,
        y_pred,
        pos_label=at_risk_label,
        zero_division=0
    )

    at_risk_f1 = f1_score(
        y_true,
        y_pred,
        pos_label=at_risk_label,
        zero_division=0
    )

    return {
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "At-Risk Precision": at_risk_precision,
        "At-Risk Recall": at_risk_recall,
        "At-Risk F1": at_risk_f1,
    }


def regression_cross_validation(
    model,
    X,
    y,
    cv=5
):
    """
    Evaluate a regression model using k-fold cross-validation.

    Returns
    -------
    dict
        Mean and standard deviation of RMSE.
    """

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error"
    )

    rmse_scores = -scores

    return {
        "CV RMSE Mean": rmse_scores.mean(),
        "CV RMSE Std": rmse_scores.std(),
        "CV RMSE Scores": rmse_scores,
    }


def classification_cross_validation(
    model,
    X,
    y,
    cv=5
):
    """
    Evaluate a classification model using k-fold
    cross-validation with Macro F1.
    """

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1_macro"
    )

    return {
        "CV Macro F1 Mean": scores.mean(),
        "CV Macro F1 Std": scores.std(),
        "CV Macro F1 Scores": scores,
    }


def get_classification_report(
    y_true,
    y_pred
):
    """
    Generate the detailed classification report.
    """

    return classification_report(
        y_true,
        y_pred,
        target_names=[
            "Not At Risk",
            "At Risk"
        ],
        zero_division=0
    )


def compare_regression_models(
    models,
    X,
    y,
    cv=5
):
    """
    Compare multiple regression models using cross-validation.
    """

    results = []

    for name, model in models.items():

        cv_results = regression_cross_validation(
            model,
            X,
            y,
            cv=cv
        )

        results.append(
            {
                "Model": name,
                "CV RMSE": cv_results["CV RMSE Mean"],
                "CV RMSE Std": cv_results["CV RMSE Std"],
            }
        )

    return results


def compare_classification_models(
    models,
    X,
    y,
    cv=5
):
    """
    Compare multiple classification models using
    Macro F1 cross-validation.
    """

    results = []

    for name, model in models.items():

        cv_results = classification_cross_validation(
            model,
            X,
            y,
            cv=cv
        )

        results.append(
            {
                "Model": name,
                "CV Macro F1": cv_results["CV Macro F1 Mean"],
                "CV Macro F1 Std": cv_results["CV Macro F1 Std"],
            }
        )

    return results


if __name__ == "__main__":

    print("Evaluation module loaded successfully.")

    print("\nAvailable functions:")

    print("- regression_metrics")
    print("- classification_metrics")
    print("- regression_cross_validation")
    print("- classification_cross_validation")
    print("- get_classification_report")
    print("- compare_regression_models")
    print("- compare_classification_models")

    print("\nEvaluation module test completed successfully.")
