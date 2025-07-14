from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import trange

from propensity_estimation import get_propensity_estimator


def read_causal_data(
    path: str = "../data/eqls_processed.csv",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load covariates *X*, treatment indicator *T* and outcome *Y*.

    The CSV is expected to contain *treatment* and *outcome* columns.
    All remaining columns are used as covariates.
    """
    data: DataFrame = pd.read_csv(path)
    if not {"treatment", "outcome"}.issubset(data.columns):
        raise ValueError("CSV must contain 'treatment' and 'outcome' columns")

    X: DataFrame = data.drop(columns=["treatment", "outcome"])
    T: Series = data["treatment"].astype(int)
    Y: Series = data["outcome"].astype(float)
    print(f"[+] Loaded data with shapes: X={X.shape}, T={T.shape}, Y={Y.shape}")
    return X, T, Y


class BaseATEEstimator:
    """Abstract base class — all concrete estimators implement :py:meth:`estimate_ate`."""

    def estimate_ate(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> float:
        """Return the Average Treatment Effect (ATE)."""
        raise NotImplementedError


@dataclass
class SLearnerATE(BaseATEEstimator):
    """S‑learner: single model with treatment indicator appended as a feature."""

    outcome_model = RandomForestClassifier(n_estimators=200)

    def estimate_ate(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> float:
        model = clone(self.outcome_model)
        # Append treatment indicator as an extra column
        X_aug: DataFrame = X.copy()
        X_aug["T"] = T.values
        model.fit(X_aug, Y)

        # Counter‑factual predictions
        X_treated: DataFrame = X.copy()
        X_treated["T"] = 1

        X_control: DataFrame = X.copy()
        X_control["T"] = 0

        mu1 = model.predict_proba(X_treated)[:, 1]
        mu0 = model.predict_proba(X_control)[:, 1]
        ite = mu1 - mu0
        return float(np.mean(ite))


@dataclass
class TLearnerATE(BaseATEEstimator):
    """T‑learner: separate models for treated and control groups."""

    outcome_model = RandomForestClassifier(n_estimators=200)

    def estimate_ate(
        self, X: pd.DataFrame, T: pd.Series, Y: pd.Series
    ) -> float:  # noqa: D401
        model_treated = clone(self.outcome_model)
        model_control = clone(self.outcome_model)

        model_treated.fit(X[T == 1], Y[T == 1])
        model_control.fit(X[T == 0], Y[T == 0])

        mu1 = model_treated.predict_proba(X)[:, 1]
        mu0 = model_control.predict_proba(X)[:, 1]
        ite = mu1 - mu0
        return float(np.mean(ite))


@dataclass
class IPWATE(BaseATEEstimator):
    """Inverse‑Probability‑Weighted estimator."""

    propensity_model = get_propensity_estimator()

    def estimate_ate(
        self, X: pd.DataFrame, T: pd.Series, Y: pd.Series
    ) -> float:  # noqa: D401
        model = clone(self.propensity_model)
        model.fit(X, T)
        p = model.predict_proba(X)[:, 1].clip(1e-4, 1 - 1e-4)  # avoid div‑by‑zero
        weights = np.where(T == 1, 1 / p, 1 / (1 - p))
        return float(np.average(Y * (2 * T - 1), weights=weights))

    def propensity_scores(
        self, X: pd.DataFrame, T: pd.Series
    ) -> np.ndarray:  # helper for overlap plot
        """Return propensity scores after fitting the internal model."""
        model = clone(self.propensity_model)
        model.fit(X, T)
        return model.predict_proba(X)[:, 1]


def bootstrap_ci(
    estimator: BaseATEEstimator,
    X: pd.DataFrame,
    T: pd.Series,
    Y: pd.Series,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = 42,
) -> Tuple[float, Tuple[float, float], np.ndarray]:
    """Compute percentile bootstrap CI for an ATE estimator."""
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(X))
    samples = np.empty(n_bootstraps)

    for b in trange(n_bootstraps):
        draw = rng.choice(idx, size=len(idx), replace=True)
        X_b, T_b, Y_b = X.iloc[draw], T.iloc[draw], Y.iloc[draw]
        samples[b] = estimator.estimate_ate(X_b, T_b, Y_b)

    lower, upper = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point_estimate = estimator.estimate_ate(X, T, Y)
    return point_estimate, (lower, upper), samples


def plot_ate_violin(ax: plt.Axes, samps: Dict[str, np.ndarray]) -> None:
    """Violin plot of bootstrap distributions."""
    positions = np.arange(1, len(samps) + 1)
    ax.violinplot([samps[k] for k in samps], positions=positions, showextrema=False)
    ax.set_xticks(positions, labels=list(samps.keys()), rotation=15)
    ax.set_ylabel("ATE")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    # overlay point estimates & CIs
    for pos, key in zip(positions, samps):
        pe = np.mean(samps[key])
        ci_low, ci_high = np.percentile(samps[key], [2.5, 97.5])
        ax.errorbar(
            pos, pe, yerr=[[pe - ci_low], [ci_high - pe]], fmt="o", color="darkred"
        )
    ax.set_title("Bootstrap distributions of ATE estimates")


def plot_propensity_overlap(ax: plt.Axes, scores: np.ndarray, T: pd.Series) -> None:
    """Histogram overlay of propensity scores for treated vs control."""
    ax.hist(scores[T == 1], bins=30, alpha=0.6, label="Treated", density=True)
    ax.hist(scores[T == 0], bins=30, alpha=0.6, label="Control", density=True)
    ax.set_xlabel("Propensity score")
    ax.set_ylabel("Density")
    ax.set_title("Propensity score overlap")
    ax.legend()


def main() -> None:
    plt.style.use("ggplot")

    X, T, Y = read_causal_data()

    estimators: Dict[str, BaseATEEstimator] = {
        "S‑learner": SLearnerATE(),
        "T‑learner": TLearnerATE(),
        "IPW": IPWATE(),
    }

    results: Dict[str, Tuple[float, Tuple[float, float]]] = {}
    distributions: Dict[str, np.ndarray] = {}

    for name, est in estimators.items():
        pe, ci, samps = bootstrap_ci(est, X, T, Y, n_bootstraps=30, alpha=0.05)
        results[name] = (pe, ci)
        distributions[name] = samps
        print(f"{name:<10}  ATE = {pe: .4f}  95% CI = [{ci[0]: .4f}, {ci[1]: .4f}]")

    ipw_model: IPWATE = estimators["IPW"]
    prop_scores = ipw_model.propensity_scores(X, T)

    _, (ax_viol, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))
    plot_ate_violin(ax_viol, distributions)
    plot_propensity_overlap(ax_hist, prop_scores, T)
    plt.tight_layout()
    plt.savefig("../figs/ate_violin_and_overlap.svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
