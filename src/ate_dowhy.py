"""ATE estimation using DoWhy + custom estimators with proper numeric/categorical pipelining.

Implements:
 - Identification with the project's causal graph (`graphs/full_causal.gpickle`).
 - DoWhy estimators: propensity score matching, weighting, stratification, regression.
 - Custom estimators: nearest-neighbor matching (propensity and Mahalanobis), IPW, AIPW.
 - Bootstrap confidence intervals for all estimators.

Run example (quick):
    python3 -m src.ate_dowhy --bootstrap 20 --no-plots

Outputs:
 - data/ate_results_dowhy.csv
 - figs/propensity_overlap_dowhy.svg
 - figs/ate_bootstrap_distribution_dowhy.svg
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dowhy import CausalModel
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


ROOT = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT, "../data/eqls_processed.csv")
DICT_PATH = os.path.join(ROOT, "../data/dictionary.json")
GRAPH_PATH = os.path.join(ROOT, "../graphs/full_causal.gpickle")
FIG_DIR = os.path.join(ROOT, "../figs")
RESULTS_PATH = os.path.join(ROOT, "../data/ate_results_dowhy.csv")

TREATMENT = "Y11_Q57"
OUTCOME = "Y11_MWIndex"


def load_dictionary() -> dict:
    with open(DICT_PATH, "r") as f:
        return json.load(f)


def load_graph() -> nx.DiGraph:
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def load_data(graph: nx.Graph | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # ensure treatment binary
    df[TREATMENT] = (df[TREATMENT] > 0.5).astype(int)
    if graph is not None:
        cols = [n for n in graph.nodes if n in df.columns]
        df = df[cols].copy()
    return df


def identify_feature_types(df: pd.DataFrame, dictionary: dict) -> Tuple[List[str], List[str]]:
    num = []
    cat = []
    for c in df.columns:
        if c in (TREATMENT, OUTCOME):
            continue
        meta = dictionary.get(c, {})
        dtype = meta.get("data_type", "unknown")
        if dtype in {"numeric_continuous", "numeric_discrete", "ordinal"}:
            num.append(c)
        elif dtype == "categorical_nominal":
            cat.append(c)
        elif dtype == "categorical_binary":
            # map binary to 0/1
            if df[c].nunique() == 2:
                vals = sorted(df[c].dropna().unique())
                mapping = {vals[0]: 0, vals[-1]: 1}
                df[c] = df[c].map(mapping)
                num.append(c)
            else:
                cat.append(c)
        else:
            if pd.api.types.is_numeric_dtype(df[c]):
                num.append(c)
            else:
                cat.append(c)
    return num, cat


def build_pipeline(df: pd.DataFrame, dictionary: dict):
    num_cols, cat_cols = identify_feature_types(df, dictionary)
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    ct = ColumnTransformer(transformers, remainder="drop")
    pipe = Pipeline([("ct", ct)])
    X = pd.DataFrame(pipe.fit_transform(df.drop(columns=[TREATMENT, OUTCOME])))
    return pipe, X, num_cols, cat_cols


# Custom estimators (IPW, Matching, AIPW) reusing pipeline output

def fit_propensity(X: pd.DataFrame, t: pd.Series):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, t)
    p = model.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return p, model


def estimate_ipw(y: np.ndarray, t: np.ndarray, p: np.ndarray) -> float:
    pt = t.mean()
    sw = (pt ** t) * ((1 - pt) ** (1 - t)) / (p ** t) / ((1 - p) ** (1 - t))
    y1 = np.sum(sw * t * y) / np.sum(sw * t)
    y0 = np.sum(sw * (1 - t) * y) / np.sum(sw * (1 - t))
    return y1 - y0


def nearest_neighbor_propensity_matching(p: np.ndarray, t: np.ndarray, y: np.ndarray, caliper: float = 0.2) -> float:
    logit = np.log(p / (1 - p))
    sd = np.std(logit)
    cal = caliper * sd
    treated = np.where(t == 1)[0]
    control = np.where(t == 0)[0]
    p_t = p[treated][:, None]
    p_c = p[control][None, :]
    d = np.abs(p_t - p_c)
    idx = d.argmin(axis=1)
    min_d = d.min(axis=1)
    valid = min_d <= cal
    if valid.sum() == 0:
        return float('nan')
    matched_t = treated[valid]
    matched_c = control[idx[valid]]
    return y[matched_t].mean() - y[matched_c].mean()


def aipw(y: np.ndarray, t: np.ndarray, p: np.ndarray, X: np.ndarray) -> float:
    reg_t = GradientBoostingRegressor(random_state=0)
    reg_c = GradientBoostingRegressor(random_state=0)
    reg_t.fit(X[t == 1], y[t == 1])
    reg_c.fit(X[t == 0], y[t == 0])
    mu1 = reg_t.predict(X)
    mu0 = reg_c.predict(X)
    est = np.mean((t * (y - mu1) / p) - ((1 - t) * (y - mu0) / (1 - p)) + (mu1 - mu0))
    return est


def plot_propensity_overlap(p: np.ndarray, t: np.ndarray, suffix: str = "dowhy"):
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.kdeplot(p[t == 1], label="T=1", fill=True, alpha=0.4)
    sns.kdeplot(p[t == 0], label="T=0", fill=True, alpha=0.4)
    plt.xlabel('Propensity')
    plt.ylabel('Density')
    plt.title('Propensity overlap')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'propensity_overlap_{suffix}.svg'))
    plt.close()


def plot_bootstrap(df_boot: pd.DataFrame, suffix: str = "dowhy"):
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure(figsize=(8, 4))
    dfm = df_boot.melt(var_name='method', value_name='ate')
    sns.violinplot(data=dfm, x='method', y='ate', inner='quartile', cut=0)
    plt.axhline(0, color='k', ls='--')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'ate_bootstrap_distribution_{suffix}.svg'))
    plt.close()


def run_dowhy_estimators(df: pd.DataFrame, graph: nx.Graph, methods: List[str], bootstrap: int, seed: int, no_plots: bool):
    model = CausalModel(data=df, treatment=TREATMENT, outcome=OUTCOME, graph=nx.nx_pydot.to_pydot(graph).to_string())
    estimand = model.identify_effect()
    results: Dict[str, float] = {}
    boot_records: List[Dict[str, float]] = []

    # Build pipeline for custom estimators
    dictionary = load_dictionary()
    pipe, X_proc, num_cols, cat_cols = build_pipeline(df, dictionary)
    y = df[OUTCOME].values
    t = df[TREATMENT].values
    p_hat, _ = fit_propensity(X_proc, df[TREATMENT])

    # point estimates via DoWhy for selected built-in methods
    for m in methods:
        try:
            est = model.estimate_effect(estimand, method_name=m)
            val = float(est.value)
            results[m] = val
        except Exception as e:
            results[m] = float('nan')

    # custom estimators
    try:
        results['ipw_custom'] = estimate_ipw(y, t, p_hat)
    except Exception:
        results['ipw_custom'] = float('nan')
    try:
        results['matching_propensity_custom'] = nearest_neighbor_propensity_matching(p_hat, t, y)
    except Exception:
        results['matching_propensity_custom'] = float('nan')
    try:
        results['aipw_custom'] = aipw(y, t, p_hat, X_proc.values)
    except Exception:
        results['aipw_custom'] = float('nan')

    # Bootstrap (re-run custom estimators and Dowhy's lightweight methods)
    rng = np.random.default_rng(seed)
    n = len(df)
    for b in tqdm(range(bootstrap), desc='Bootstrap'):  # type: ignore
        idx = rng.integers(0, n, size=n)
        df_b = df.iloc[idx].reset_index(drop=True)
        # Dowhy estimators: re-fit and estimate for methods
        model_b = CausalModel(data=df_b, treatment=TREATMENT, outcome=OUTCOME, graph=nx.nx_pydot.to_pydot(graph).to_string())
        estimand_b = model_b.identify_effect()
        rec: Dict[str, float] = {}
        # custom pipeline
        pipe_b, Xb_proc, _, _ = build_pipeline(df_b, dictionary)
        yb = df_b[OUTCOME].values
        tb = df_b[TREATMENT].values
        try:
            pb, _ = fit_propensity(Xb_proc, df_b[TREATMENT])
        except Exception:
            pb = np.clip(np.full(len(yb), tb.mean()), 1e-3, 1 - 1e-3)
        # DoWhy methods
        for m in methods:
            try:
                est_b = model_b.estimate_effect(estimand_b, method_name=m)
                rec[m] = float(est_b.value)
            except Exception:
                rec[m] = float('nan')
        # custom
        try:
            rec['ipw_custom'] = estimate_ipw(yb, tb, pb)
        except Exception:
            rec['ipw_custom'] = float('nan')
        try:
            rec['matching_propensity_custom'] = nearest_neighbor_propensity_matching(pb, tb, yb)
        except Exception:
            rec['matching_propensity_custom'] = float('nan')
        try:
            rec['aipw_custom'] = aipw(yb, tb, pb, Xb_proc.values)
        except Exception:
            rec['aipw_custom'] = float('nan')
        boot_records.append(rec)

    df_boot = pd.DataFrame(boot_records)

    # Summaries
    summary_rows = []
    for col in df_boot.columns:
        vals = df_boot[col].dropna()
        if len(vals) == 0:
            continue
        ate = vals.mean()
        se = vals.std(ddof=1)
        lower, upper = np.percentile(vals, [2.5, 97.5])
        summary_rows.append({'method': col, 'ate': ate, 'se': se, 'ci_lower': lower, 'ci_upper': upper, 'n_boot': len(vals)})

    summary = pd.DataFrame(summary_rows).sort_values('method')
    summary.to_csv(RESULTS_PATH, index=False)

    if not no_plots:
        plot_propensity_overlap(p_hat, t, suffix='dowhy')
        plot_bootstrap(df_boot, suffix='dowhy')

    return results, summary, df_boot


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(list(argv) if argv is not None else None)

    graph = load_graph()
    df = load_data(graph)
    # methods to try via DoWhy
    methods = [
        'backdoor.propensity_score_matching',
        'backdoor.propensity_score_weighting',
        'backdoor.propensity_score_stratification',
        'backdoor.linear_regression'
    ]
    results, summary, df_boot = run_dowhy_estimators(df, graph, methods, args.bootstrap, args.seed, args.no_plots)

    print('\nPoint estimates (DoWhy + custom):')
    for k, v in results.items():
        print(f'  {k:<35} {v:.4f}')

    print('\nBootstrap summary saved to:', RESULTS_PATH)
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
