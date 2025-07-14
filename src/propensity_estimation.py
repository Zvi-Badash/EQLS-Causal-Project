from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import dtype, ndarray
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from gbdt_utils import GBDTClassifier


def read_data_for_propensity() -> tuple[pd.DataFrame, pd.Series]:
    """
    Reads the data from a CSV file and returns a DataFrame.
    """
    data = pd.read_csv("../data/eqls_processed.csv").drop(columns=["outcome"])
    return data.drop(columns=["treatment"]), data["treatment"]


def prepare_for_model_selection(
    X: pd.DataFrame, y: pd.Series, frac_test: float = 0.2
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits the data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=frac_test, random_state=42
    )
    return X_train, y_train, X_test, y_test


def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float | ndarray[tuple[Any, ...], dtype[Any]]]:
    """
    Evaluates the model using Brier score, F1 score, and ROC AUC score.
    """
    y_pred: np.ndarray = model.predict(X_test)
    y_proba: np.ndarray = model.predict_proba(X_test)[:, 1]

    brier_score: float = brier_score_loss(y_test, y_proba)
    f1: float = f1_score(y_test, y_pred)
    roc_auc: float = roc_auc_score(y_test, y_proba)
    cc: np.ndarray = calibration_curve(y_test, y_proba, n_bins=10)

    return {
        "brier_score": brier_score,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "calibration_curve": cc,
    }


def get_propensity_estimator() -> GBDTClassifier:
    return (
        GBDTClassifier(metric_key="roc_auc", do_tune=True, frac_features_keep=0.9)
        .fit(*read_data_for_propensity())
        .freeze()
    )


def main() -> None:
    plt.style.use("ggplot")
    X, y = read_data_for_propensity()
    X_train, y_train, X_test, y_test = prepare_for_model_selection(X, y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "GBDT Classifier": GBDTClassifier(
            metric_key="roc_auc", do_tune=True, frac_features_keep=0.9
        ),
    }

    print(f"{'Model':<24} {'F1':<12} {'Brier':<12} {'ROC AUC':<12}")
    print("-" * (24 + 12 + 12 + 12))
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics: dict[str, float | ndarray[tuple[Any, ...], dtype[Any]]] = (
            evaluate_model(model, X_test, y_test)
        )
        calibration_curve: float | ndarray[tuple[Any, ...], dtype[Any]] = metrics.pop(
            "calibration_curve"
        )

        print(
            f"{name:<24}"
            f"{metrics['f1_score']:<12.3f} "
            f"{metrics['brier_score']:<12.3f} "
            f"{metrics['roc_auc']:<12.3f} "
        )

        plt.plot(calibration_curve[1], calibration_curve[0], marker="o", label=name)
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title(f"Calibration Curve for {name}")
    plt.legend()
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    plt.savefig("../figs/calibration_curves.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
