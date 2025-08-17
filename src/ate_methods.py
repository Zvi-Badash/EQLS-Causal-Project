"""ATE estimation methods (Matching, IPW, AIPW) with bootstrapped CIs.

This script follows causal presentation guidelines by:
 - Using proper variable type handling based on `data/dictionary.json` metadata
 - Estimating multiple causal estimators (propensity score matching, IPW, AIPW)
 - Providing bootstrap confidence intervals with reproducibility
 - Emitting diagnostics (propensity overlap, balance) and distribution plots

Run (default 200 bootstrap replications):
    python -m src.ate_methods --bootstrap 200

Outputs:
 - data/ate_results.csv : summary table of ATE estimates + 95% CIs
 - figs/propensity_overlap.svg : propensity score overlap by treatment group
 - figs/ate_bootstrap_distribution.svg : distribution of bootstrapped ATEs
 - Console printout of balance diagnostics (standardized mean differences)

Assumptions:
 - Processed dataset at data/eqls_processed.csv with binary treatment `Y11_Q57`
 - Outcome is continuous wellbeing index `Y11_MWIndex`
 - Remaining columns are covariates (some may not appear in dictionary if trimmed)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:  # fallback noop progress
    def tqdm(x, **kwargs):  # type: ignore
        return x


DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/eqls_processed.csv")
DICT_PATH = os.path.join(os.path.dirname(__file__), "../data/dictionary.json")
FIG_DIR = os.path.join(os.path.dirname(__file__), "../figs")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "../data/ate_results.csv")

TREATMENT_COL = "Y11_Q57"
OUTCOME_COL = "Y11_MWIndex"


@dataclass
class ATEEstimate:
    method: str
    ate: float
    se: float | None
    ci_lower: float | None
    ci_upper: float | None
    extra: Dict[str, float] | None = None


def load_dictionary() -> dict:
    with open(DICT_PATH, "r") as f:
        return json.load(f)


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at {DATA_PATH}. Run preprocessing first.")
    df = pd.read_csv(DATA_PATH)
    if TREATMENT_COL not in df.columns or OUTCOME_COL not in df.columns:
        raise ValueError("Expected treatment and outcome columns not found in processed dataset.")
    df[TREATMENT_COL] = (df[TREATMENT_COL] > 0.5).astype(int)
    return df


def identify_feature_types(df: pd.DataFrame, dictionary: dict) -> Tuple[List[str], List[str]]:
    numeric_like: List[str] = []
    categorical_nominal: List[str] = []
    for col in df.columns:
        if col in (TREATMENT_COL, OUTCOME_COL):
            continue
        meta = dictionary.get(col, {})
        dtype = meta.get("data_type", "unknown")
        if dtype in {"numeric_continuous", "numeric_discrete", "ordinal"}:
            numeric_like.append(col)
        elif dtype == "categorical_nominal":
            categorical_nominal.append(col)
        elif dtype == "categorical_binary":
            if df[col].nunique() == 2:
                vals = sorted(df[col].dropna().unique())
                mapping = {vals[0]: 0, vals[-1]: 1}
                df[col] = df[col].map(mapping)
                numeric_like.append(col)
            else:
                categorical_nominal.append(col)
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_like.append(col)
            else:
                categorical_nominal.append(col)
    return numeric_like, categorical_nominal


def build_design_matrix(df: pd.DataFrame, dictionary: dict):
    t = df[TREATMENT_COL].astype(int)
    y = df[OUTCOME_COL].astype(float)
    X_raw = df.drop(columns=[TREATMENT_COL, OUTCOME_COL])
    num_cols, cat_cols = identify_feature_types(df, dictionary)
    num_cols = [c for c in X_raw.columns if c in num_cols]
    cat_cols = [c for c in X_raw.columns if c in cat_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False), cat_cols))
    ct = ColumnTransformer(transformers, remainder="drop")
    pipeline = Pipeline([("ct", ct)])
    X = pd.DataFrame(pipeline.fit_transform(X_raw))
    return t, y, X, pipeline


def standardized_mean_difference(X: np.ndarray, t: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    t = np.asarray(t)
    X = np.asarray(X)
    if w is None:
        w = np.ones_like(t, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
    wt = w * (t == 1)
    wc = w * (t == 0)
    mt = np.average(X, weights=wt, axis=0) / (wt.mean() / ((t == 1).mean()) if wt.mean() > 0 else 1)
    mc = np.average(X, weights=wc, axis=0) / (wc.mean() / ((t == 0).mean()) if wc.mean() > 0 else 1)
    def wvar(x, weights):
        if weights.sum() == 0:
            return np.zeros(x.shape[1])
        ave = np.average(x, axis=0, weights=weights)
        return np.average((x - ave) ** 2, axis=0, weights=weights)
    vt = wvar(X[t == 1], wt[t == 1])
    vc = wvar(X[t == 0], wc[t == 0])
    denom = np.sqrt((vt + vc) / 2 + 1e-8)
    smd = (mt - mc) / denom
    return smd


def fit_propensity(X: pd.DataFrame, t: pd.Series, random_state: int = 42):
    model = LogisticRegression(max_iter=1000, penalty="l2", solver="lbfgs")
    model.fit(X, t)
    p = model.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return p, model


def estimate_ate_ipw(y: np.ndarray, t: np.ndarray, p: np.ndarray) -> float:
    pt = t.mean()
    sw = (pt ** t) * ((1 - pt) ** (1 - t)) / (p ** t) / ((1 - p) ** (1 - t))
    y1 = np.sum(sw * t * y) / np.sum(sw * t)
    y0 = np.sum(sw * (1 - t) * y) / np.sum(sw * (1 - t))
    return y1 - y0


def nearest_neighbor_matching(p: np.ndarray, t: np.ndarray, y: np.ndarray, caliper_mult: float = 0.2) -> float:
    logit = np.log(p / (1 - p))
    caliper = caliper_mult * np.std(logit)
    treated_idx = np.where(t == 1)[0]
    control_idx = np.where(t == 0)[0]
    p_t = p[treated_idx][:, None]
    p_c = p[control_idx][None, :]
    dists = np.abs(p_t - p_c)
    matches = dists.argmin(axis=1)
    min_d = dists.min(axis=1)
    valid = min_d <= caliper
    matched_t = treated_idx[valid]
    matched_c = control_idx[matches[valid]]
    if len(matched_t) == 0:
        return float("nan")
    unique_c, counts_c = np.unique(matched_c, return_counts=True)
    control_weights = {c: w for c, w in zip(unique_c, counts_c)}
    y_t = y[matched_t].mean()
    y_c = np.average(y[matched_c], weights=[control_weights[c] for c in matched_c])
    return y_t - y_c


def estimate_ate_aipw(y: np.ndarray, t: np.ndarray, p: np.ndarray, X: np.ndarray, random_state: int = 42) -> float:
    reg_t = GradientBoostingRegressor(random_state=random_state)
    reg_c = GradientBoostingRegressor(random_state=random_state)
    reg_t.fit(X[t == 1], y[t == 1])
    reg_c.fit(X[t == 0], y[t == 0])
    mu1 = reg_t.predict(X)
    mu0 = reg_c.predict(X)
    est = np.mean((t * (y - mu1) / p) - ((1 - t) * (y - mu0) / (1 - p)) + (mu1 - mu0))
    return est


def bootstrap_ates(X: np.ndarray, y: np.ndarray, t: np.ndarray, B: int, caliper_mult: float, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    records = []
    n = len(y)
    for _ in tqdm(range(B), desc="Bootstrapping"):
        idx = rng.integers(0, n, size=n)
        X_b, y_b, t_b = X[idx], y[idx], t[idx]
        try:
            p_b, _ = fit_propensity(pd.DataFrame(X_b), pd.Series(t_b), random_state)
            ipw = estimate_ate_ipw(y_b, t_b, p_b)
            match = nearest_neighbor_matching(p_b, t_b, y_b, caliper_mult)
            aipw = estimate_ate_aipw(y_b, t_b, p_b, X_b, random_state)
            records.append({"ipw": ipw, "matching": match, "aipw": aipw})
        except Exception:
            continue
    return pd.DataFrame(records)


def summarize_bootstrap(df_boot: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for method in df_boot.columns:
        vals = df_boot[method].dropna()
        if len(vals) == 0:
            continue
        ate = vals.mean()
        se = vals.std(ddof=1)
        lower, upper = np.percentile(vals, [2.5, 97.5])
        summary_rows.append({"method": method, "ate": ate, "se": se, "ci_lower": lower, "ci_upper": upper, "n_boot": len(vals)})
    return pd.DataFrame(summary_rows)


def plot_propensity_overlap(p: np.ndarray, t: np.ndarray) -> None:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(p[t == 1], label="T=1", fill=True, alpha=0.4)
    sns.kdeplot(p[t == 0], label="T=0", fill=True, alpha=0.4)
    plt.xlabel("Propensity score")
    plt.ylabel("Density")
    plt.title("Propensity Score Overlap")
    plt.legend()
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "propensity_overlap.svg"))
    plt.close()


def plot_bootstrap_distribution(df_boot: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    df_long = df_boot.melt(var_name="method", value_name="ate")
    sns.violinplot(data=df_long, x="method", y="ate", inner="quartile", cut=0)
    plt.axhline(0, color="black", lw=1, ls="--")
    plt.xlabel("Estimator")
    plt.ylabel("ATE")
    plt.title("Bootstrap ATE Distribution")
    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIG_DIR, "ate_bootstrap_distribution.svg"))
    plt.close()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ATE estimation with multiple estimators and bootstrap CIs")
    parser.add_argument("--bootstrap", type=int, default=200, help="Number of bootstrap replications (default 200)")
    parser.add_argument("--caliper-mult", type=float, default=0.2, help="Caliper multiplier on SD(logit(p)) for matching")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    args = parser.parse_args(list(argv) if argv is not None else None)

    dictionary = load_dictionary()
    df = load_data()
    t, y, X, _ = build_design_matrix(df, dictionary)

    p, _ = fit_propensity(X, t, args.seed)

    ate_ipw = estimate_ate_ipw(y.values, t.values, p)
    ate_match = nearest_neighbor_matching(p, t.values, y.values, args.caliper_mult)
    ate_aipw = estimate_ate_aipw(y.values, t.values, p, X.values, args.seed)

    smd_unw = standardized_mean_difference(X.values, t.values)
    w_ipw = ((t.mean() ** t.values) * ((1 - t.mean()) ** (1 - t.values)) / (p ** t.values) / ((1 - p) ** (1 - t.values)))
    smd_ipw = standardized_mean_difference(X.values, t.values, w_ipw)

    print("\nPoint Estimates:")
    print(f"  IPW ATE:   {ate_ipw:.4f}")
    print(f"  Match ATE: {ate_match:.4f}")
    print(f"  AIPW ATE:  {ate_aipw:.4f}")

    print("\nBalance (mean absolute SMD of first 20 features):")
    print(f"  Unweighted: {np.abs(smd_unw)[:20].mean():.3f}")
    print(f"  IPW-weight: {np.abs(smd_ipw)[:20].mean():.3f}")

    df_boot = bootstrap_ates(X.values, y.values, t.values, args.bootstrap, args.caliper_mult, args.seed)
    summary = summarize_bootstrap(df_boot)
    print("\nBootstrap Summary (95% percentile CIs):")
    for _, row in summary.iterrows():
        print(f"  {row['method']:<9} ATE={row['ate']:.4f}  SE={row['se']:.4f}  CI=({row['ci_lower']:.4f}, {row['ci_upper']:.4f})  n_boot={int(row['n_boot'])}")

    summary.to_csv(RESULTS_PATH, index=False)
    print(f"\n[+] Saved results to {RESULTS_PATH}")

    if not args.no_plots:
        plot_propensity_overlap(p, t.values)
        plot_bootstrap_distribution(df_boot)
        print(f"[+] Saved plots to {FIG_DIR}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
