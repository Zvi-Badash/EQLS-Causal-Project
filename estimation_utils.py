import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from dowhy import CausalModel
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from tqdm.notebook import trange

from structure_data import choose_columns, preprocess_data

ROOT = Path("./")
RAW_DATA = ROOT / "raw_data/csv/eqls_2007and2011.csv"
DICT_PATH = ROOT / "data/dictionary.json"
GRAPH_PATH = ROOT / "graphs/full_causal.gpickle"
TREATMENT = "Y11_Q57"
OUTCOME = "Y11_MWIndex"


def _prep_ordinals_inplace(df: pd.DataFrame, ord_cols, known_orders=None):
    """
    Map ordinal levels to 0..m-1 and scale to [0,1] so L1 equals Gower per-feature.
    known_orders: optional dict {col: [lowest,...,highest]} to control ordering.
    """
    for c in ord_cols:
        if known_orders and c in known_orders:
            order = {v: i for i, v in enumerate(known_orders[c])}
            df[c] = df[c].map(order).astype(float)
            m = max(order.values()) if order else 1
            df[c] = df[c] / (m if m > 0 else 1.0)
        else:
            # infer: if numeric-coded, scale min–max; else sort unique labels
            if pd.api.types.is_numeric_dtype(df[c]):
                lo, hi = df[c].min(), df[c].max()
                rng = (hi - lo) if hi > lo else 1.0
                df[c] = (df[c] - lo) / rng
            else:
                levels = sorted(df[c].dropna().unique().tolist())
                mapping = {v: i for i, v in enumerate(levels)}
                m = max(mapping.values()) if mapping else 1
                df[c] = df[c].map(mapping).astype(float) / (m if m > 0 else 1.0)


def load_data() -> pd.DataFrame:
    """Load and preprocess the raw EQLS data."""
    bdvs = [
        "Y11_EmploymentStatus",
        "Y11_HHstructure",
        "Y11_HHsize",
        "Y11_Agecategory",
        "Y11_Q7",
        "Y11_Q31",
        "Y11_Country",
        "Y11_Q32",
        "Y11_HH2a",
        TREATMENT,
        OUTCOME,
    ]
    df = choose_columns()
    df = preprocess_data(
        df,
        na_threshold=0.5,
        impute_strategy="drop",
        treatment_dichotomize_value="median",
        treatment_column=TREATMENT,
        backdoor_variables=bdvs,
    )
    df.to_csv("data/eqls_processed.csv", index=False)
    return df


def _get_schema() -> dict:
    categorical = ["Y11_Q32", "Y11_Q7"]
    ordinal = [
        "Y11_Agecategory",
        "Y11_Country",
        "Y11_EmploymentStatus",
        "Y11_HH2a",
        "Y11_HHsize",
        "Y11_HHstructure",
        "Y11_Q31",
    ]
    return {"cat": categorical, "ord": ordinal}


def load_graph() -> nx.DiGraph:
    """Load the causal graph describing relationships among variables."""
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def t_learner_estimate(
    df: pd.DataFrame,
    schema: dict,
    treatment_col: str,
    outcome_col: str,
) -> float:
    """
    T-learner ATE using two outcome models:
      - f1 trained on treated data
      - f0 trained on control data
    Returns mean_i [ f1(x_i) - f0(x_i) ].
    If outcome is not binary {0,1}, falls back to linear regression.

    Assumes df already has ordinal columns numerically scaled.
    """
    # Features: confounders only (no T, no Y)
    feat_cols = schema["cat"] + schema["ord"]
    X = df[feat_cols].copy()

    # Ensure categoricals are treated as such, then one-hot on FULL X (to align cols)
    for c in schema["cat"]:
        X[c] = X[c].astype("category")
    X_enc = pd.get_dummies(X, columns=schema["cat"], drop_first=True)

    T = df[treatment_col].astype(int).values
    y = df[outcome_col].values

    X1, y1 = X_enc[T == 1], y[T == 1]
    X0, y0 = X_enc[T == 0], y[T == 0]

    g1 = LinearRegression()
    g0 = LinearRegression()
    g1.fit(X1, y1)
    g0.fit(X0, y0)
    mu1 = g1.predict(X_enc)
    mu0 = g0.predict(X_enc)

    ite = mu1 - mu0
    ate = float(np.nanmean(ite))

    return ate


def estimate_effects(
    df: pd.DataFrame,
    graph,
    return_cis: bool = False,
    ci_alpha: float = 0.05,
    n_boot: int = 10,
    random_state: int = 42,
):
    """
    Estimate ATE with several DoWhy backdoor estimators (+ custom T-learner).
    If `return_cis` is True, also bootstrap the ATEs and return CIs + the
    full bootstrap distribution as a DataFrame (one column per method).

    Returns:
        - if return_cis == False:
            results: dict[str, float]
        - if return_cis == True:
            results: dict[str, float],
            ci_df:   DataFrame with columns [method, point, lo, hi, alpha, n_boot],
            boot_df: DataFrame (shape: n_boot x n_methods) of bootstrap ATEs
    """
    methods = [
        "backdoor.propensity_score_matching",
        "backdoor.propensity_score_weighting",
        "backdoor.propensity_score_stratification",
        "backdoor.linear_regression",
        "backdoor.distance_matching",
        "backdoor.T_learner",  # custom
    ]
    kwargs = {
        "backdoor.distance_matching": dict(
            target_units="ate",
            method_params={
                "distance_metric": "minkowski",
                "p": 1,
                "num_matches_per_unit": 1,
                "exact_match_cols": _get_schema()["cat"],
            },
        )
    }

    def _estimate_once(data: pd.DataFrame) -> dict:
        """Run all estimators on a given DataFrame and return {method: ATE}."""
        d = data.copy()
        _prep_ordinals_inplace(d, _get_schema()["ord"])

        model = CausalModel(
            data=d,
            treatment=TREATMENT,
            outcome=OUTCOME,
            graph=nx.nx_pydot.to_pydot(graph).to_string(),
        )
        estimand = model.identify_effect()

        out = {}
        for m in methods:
            try:
                if m == "backdoor.T_learner":
                    out[m] = float(
                        t_learner_estimate(
                            d,
                            _get_schema(),
                            treatment_col=TREATMENT,
                            outcome_col=OUTCOME,
                        )
                    )
                else:
                    est = model.estimate_effect(
                        estimand, method_name=m, **kwargs.get(m, {})
                    )
                    out[m] = float(est.value)
            except Exception as e:
                print(f"[!] Estimation with {m} failed: {e}")
                out[m] = float("nan")
        return out

    # 1) Point estimates on the full data
    results = _estimate_once(df)

    if not return_cis:
        return results

    # 2) Bootstrap (percentile CIs), DoWhy-style: resample rows and re-estimate
    rng = np.random.default_rng(random_state)
    n = len(df)
    boot_rows = []
    for b in trange(n_boot):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        boot_df_sample = df.iloc[idx].reset_index(drop=True)
        boot_rows.append(_estimate_once(boot_df_sample))
        print(f'Bootstrap {b}/{n_boot} done.')
        pd.DataFrame(boot_rows).to_csv("bootstraps/current.csv", index=False)

    boot_df = pd.DataFrame(boot_rows)  # shape: n_boot x n_methods
    # 3) Build CI table
    lo_q = ci_alpha / 2.0
    hi_q = 1.0 - ci_alpha / 2.0
    lo = boot_df.quantile(lo_q, numeric_only=True)
    hi = boot_df.quantile(hi_q, numeric_only=True)

    ci_df = pd.DataFrame(
        {
            "method": boot_df.columns,
            "point": [results[m] for m in boot_df.columns],
            "lo": [float(lo[m]) for m in boot_df.columns],
            "hi": [float(hi[m]) for m in boot_df.columns],
            "alpha": ci_alpha,
            "n_boot": n_boot,
        }
    )

    return results, ci_df, boot_df


def propensity_overlap_graph(
    df,
    graph,
    method="backdoor.propensity_score_weighting",
    ps_model=None,
    figsize=(10, 6),
    return_scores=False,
    figpath=None,
):
    """
    Fits a propensity-based DoWhy estimator, extracts propensity scores,
    and plots treated vs control distributions to check overlap.
    Returns the fitted sklearn propensity model (and optionally the scores).
    """
    df2 = df.copy()
    _prep_ordinals_inplace(df2, _get_schema()["ord"])

    model = CausalModel(
        data=df2,
        treatment=TREATMENT,
        outcome=OUTCOME,
        graph=nx.nx_pydot.to_pydot(graph).to_string(),
    )
    estimand = model.identify_effect()

    # Use logistic regression by default (any classifier with fit/predict_proba works in DoWhy) :contentReference[oaicite:2]{index=2}
    method_params = {}
    if method.startswith("backdoor.propensity_score"):
        method_params = {
            "propensity_score_model": ps_model or LogisticRegression(max_iter=1000),
            "propensity_score_column": "propensity_score",
        }

    estimate = model.estimate_effect(
        estimand, method_name=method, method_params=method_params
    )

    ps = np.asarray(estimate.propensity_scores)
    tmask = (df2[TREATMENT] == 1).to_numpy()
    plt.figure(figsize=figsize)
    plt.rcParams.update({"font.size": 14})
    plt.tight_layout()
    sns.kdeplot(ps[tmask], label="Treated", fill=True, bw_adjust=0.5)
    sns.kdeplot(ps[~tmask], label="Control", fill=True, bw_adjust=0.5)
    plt.title("")
    plt.xlabel("Propensity Score")
    plt.xlim(0, 1)
    plt.ylabel("Density")
    plt.legend()
    if figpath:
        plt.savefig(figpath, format="svg", bbox_inches="tight")
    plt.show()

    # quick numeric common-support summary
    lo = max(ps[tmask].min(), ps[~tmask].min())
    hi = min(ps[tmask].max(), ps[~tmask].max())
    frac_t = np.mean((ps[tmask] >= lo) & (ps[tmask] <= hi))
    frac_c = np.mean((ps[~tmask] >= lo) & (ps[~tmask] <= hi))
    print(
        f"Common support approx in [{lo:.3f}, {hi:.3f}]. "
        f"Coverage — Treated: {frac_t:.3f}, Control: {frac_c:.3f}"
    )

    # return the fitted sklearn model (useful for calibration/diagnostics)
    fitted_propensity_model = estimate.estimator.propensity_score_model
    return (fitted_propensity_model, ps) if return_scores else fitted_propensity_model
