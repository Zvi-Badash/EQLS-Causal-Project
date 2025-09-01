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



def matching_ate(
    df: pd.DataFrame,
    cat_cols,
    ord_cols,
    treat_col: str,
    outcome_col: str,
    dropna: bool = True,
    return_details: bool = False,
    # Ordinals binning (from previous version)
    ord_binning=None,
    # NEW: treat ordinals as exact or continuous
    ord_mode: str = "exact",  # "exact" | "continuous"
    # NEW: distance matching options when ord_mode="continuous"
    distance_metric: str = "euclidean",  # "euclidean"|"minkowski"|"mahalanobis"|"seuclidean"
    p: int = 2,                           # for minkowski
    standardize: bool = True,             # z-score for euclidean/minkowski
    num_matches_per_unit: int = 1,        # k-NN
    with_replacement: bool = True,        # candidate reuse
    caliper: float = None                 # max allowed distance (None = no caliper)
):
    """
    ATE via:
      - Exact matching on cat+ord (ord may be pre-binned via 'ord_binning'), OR
      - Mixed matching: exact on categorical, distance-based on ordinal ('continuous').

    When ord_mode="continuous":
      • Within each categorical stratum, match each unit to k nearest neighbors from the opposite
        treatment group using the chosen distance on 'ord_cols'. 
      • Impute counterfactuals from neighbors' outcomes; compute unit-level effects; ATE is the mean
        over all units with at least one valid neighbor.
    """
    if ord_mode not in {"exact", "continuous"}:
        raise ValueError("ord_mode must be 'exact' or 'continuous'.")

    covs = list(cat_cols) + list(ord_cols)
    use_cols = covs + [treat_col, outcome_col]
    data = df[use_cols].copy()

    # Optional binning of ordinals (for ord_mode='exact' or if you just want coarsening)
    if ord_binning is not None and len(ord_cols) > 0:
        _apply_ordinals_binning_inplace(data, ord_cols, ord_binning)

    if dropna:
        data = data.dropna(subset=use_cols)

    if data.empty:
        raise ValueError("No data left after NA handling.")
    unique_t = set(np.unique(data[treat_col]))
    if unique_t - {0, 1}:
        raise ValueError(f"'{treat_col}' must be binary (0/1).")

    # --- Path 1: Exact matching on all covariates (cat + ord) ---
    if ord_mode == "exact":
        grp = data.groupby(covs + [treat_col], dropna=False)
        stats = grp[outcome_col].agg(['mean', 'count']).reset_index()
        wide = stats.pivot_table(index=covs, columns=treat_col, values=['mean', 'count'])

        mean0 = wide['mean'].get(0)
        mean1 = wide['mean'].get(1)
        n0    = wide['count'].get(0)
        n1    = wide['count'].get(1)

        strata = pd.DataFrame(index=wide.index).assign(
            n0 = n0.fillna(0).astype(int),
            n1 = n1.fillna(0).astype(int),
            y0 = mean0,
            y1 = mean1,
        )

        strata = strata[(strata['n0'] > 0) & (strata['n1'] > 0)].copy()
        if strata.empty:
            raise ValueError("No overlapping strata found (no exact matches).")

        strata['N'] = strata['n0'] + strata['n1']
        strata['diff'] = strata['y1'] - strata['y0']

        total_matched = strata['N'].sum()
        ate = (strata['diff'] * strata['N']).sum() / total_matched

        if not return_details:
            return float(ate)
        details = {
            'matched_fraction': float(total_matched) / float(len(data)),
            'n_matched': int(total_matched),
            'n_total': int(len(data)),
            'mode': 'exact',
            'strata_summary': strata.reset_index(),
        }
        return float(ate), details

    # --- Path 2: Mixed mode — exact on categorical, distance on ordinals ---
    # Validate ordinals numeric for continuous distances
    if len(ord_cols) == 0:
        raise ValueError("ord_mode='continuous' requires non-empty ord_cols.")
    if not all(np.issubdtype(data[c].dtype, np.number) for c in ord_cols):
        raise TypeError("All ord_cols must be numeric for ord_mode='continuous'.")

    # Group by categorical exact strata; distance-match within each stratum
    if len(cat_cols) == 0:
        strata_groups = [("", data)]  # single pseudo-stratum
    else:
        strata_groups = list(data.groupby(cat_cols, dropna=False))

    # Collect unit-level effects
    unit_effects = []
    unmatched_units = 0
    total_units = len(data)

    for _, df_s in strata_groups:
        t_mask = df_s[treat_col] == 1
        c_mask = ~t_mask
        if (t_mask.sum() == 0) or (c_mask.sum() == 0):
            # no opposite group in this categorical stratum—nobody can be matched here
            unmatched_units += len(df_s)
            continue

        Xt = df_s.loc[t_mask, ord_cols].to_numpy(dtype=float, copy=False)
        Xc = df_s.loc[c_mask, ord_cols].to_numpy(dtype=float, copy=False)
        yt = df_s.loc[t_mask, outcome_col].to_numpy(dtype=float, copy=False)
        yc = df_s.loc[c_mask, outcome_col].to_numpy(dtype=float, copy=False)

        # Compute distance matrices between treated and controls
        Dt, Dc = _pairwise_distances_two_way(
            Xt, Xc,
            metric=distance_metric,
            p=p,
            standardize=standardize
        )
        # Optional caliper filtering
        if caliper is not None:
            Dt = Dt.copy()
            Dc = Dc.copy()
            Dt[Dt > caliper] = np.inf
            Dc[Dc > caliper] = np.inf

        # Match treated -> controls
        cf0_treated, matched_t = _impute_counterfactuals_from_neighbors(
            opposite_outcomes=yc,
            D=Dt,
            k=num_matches_per_unit,
            with_replacement=with_replacement
        )
        # Match controls -> treated
        cf1_controls, matched_c = _impute_counterfactuals_from_neighbors(
            opposite_outcomes=yt,
            D=Dc,
            k=num_matches_per_unit,
            with_replacement=with_replacement
        )

        # Build unit-level effects
        # Treated units: tau_i = y_i - E[y0 | matches]
        if matched_t.any():
            unit_effects.append(yt[matched_t] - cf0_treated[matched_t])
        # Control units: tau_j = E[y1 | matches] - y_j
        if matched_c.any():
            unit_effects.append(cf1_controls[matched_c] - yc[matched_c])

        unmatched_units += (~matched_t).sum() + (~matched_c).sum()

    if len(unit_effects) == 0:
        raise ValueError("No units could be matched under the chosen settings (check caliper/metric).")

    unit_effects = np.concatenate(unit_effects, axis=0)
    ate = float(np.mean(unit_effects))

    if not return_details:
        return ate

    details = {
        'matched_fraction': 1.0 - (unmatched_units / (2 * total_units)),  # approx: both directions counted
        'n_total': int(total_units),
        'n_effects': int(unit_effects.size),
        'mode': 'continuous',
        'distance_metric': distance_metric,
        'standardize': standardize,
        'num_matches_per_unit': num_matches_per_unit,
        'with_replacement': with_replacement,
        'caliper': caliper,
    }
    return ate, details


# ---------- Helpers ----------

def _apply_ordinals_binning_inplace(df: pd.DataFrame, ord_cols, ord_binning):
    if isinstance(ord_binning, int):
        cfg = {c: {"method": "quantile", "n_bins": ord_binning} for c in ord_cols}
    elif isinstance(ord_binning, dict):
        cfg = ord_binning
    else:
        raise ValueError("ord_binning must be None, int, or dict.")

    for c in ord_cols:
        if c not in df.columns:
            raise KeyError(f"Ordinal column '{c}' not in DataFrame.")
        spec = cfg.get(c, None)
        if spec is None:
            continue
        method = spec.get("method", "quantile")
        if method == "quantile":
            n_bins = int(spec.get("n_bins", 4))
            df[c] = pd.qcut(df[c], q=n_bins, labels=False, duplicates="drop")
        elif method == "uniform":
            n_bins = int(spec.get("n_bins", 4))
            df[c] = pd.cut(df[c], bins=n_bins, labels=False, include_lowest=True)
        elif method == "custom":
            bins = spec.get("bins", None)
            if not bins or len(bins) < 2:
                raise ValueError(f"Custom binning for '{c}' requires a 'bins' list with >=2 edges.")
            df[c] = pd.cut(df[c], bins=bins, labels=False, include_lowest=True)
        else:
            raise ValueError(f"Unknown binning method '{method}' for column '{c}'.")
        # ensure integer-ish for exact mode friendliness
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype('Int64')
        else:
            df[c] = df[c].astype('Int64')


def _zscore(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def _pairwise_distances_two_way(Xt, Xc, metric="euclidean", p=2, standardize=True):
    """
    Returns (Dt, Dc):
      Dt: distances from each treated (rows) to each control (cols)
      Dc: distances from each control (rows) to each treated (cols)
    """
    Xtp = Xt
    Xcp = Xc

    if metric in {"euclidean", "minkowski"} and standardize:
        Xtp, mu, sd = _zscore(Xt)
        Xcp = (Xc - mu) / sd

    if metric == "euclidean" or (metric == "minkowski" and p == 2):
        # squared distances then sqrt for stability
        Dt = np.sqrt(np.maximum(0.0, _cdist_sq(Xtp, Xcp)))
        Dc = Dt.T.copy()
        return Dt, Dc

    if metric == "minkowski":
        Dt = _cdist_minkowski(Xtp, Xcp, p=p)
        Dc = Dt.T.copy()
        return Dt, Dc

    if metric == "seuclidean":
        # standardized Euclidean using pooled variances
        _, mu, sd = _zscore(np.vstack([Xt, Xc]))
        Xtp = (Xt - mu) / sd
        Xcp = (Xc - mu) / sd
        Dt = np.sqrt(np.maximum(0.0, _cdist_sq(Xtp, Xcp)))
        Dc = Dt.T.copy()
        return Dt, Dc

    if metric == "mahalanobis":
        # pooled covariance inverse
        X = np.vstack([Xt, Xc])
        Xc_mu = np.mean(X, axis=0)
        Xc_centered = X - Xc_mu
        S = np.cov(Xc_centered, rowvar=False)
        # regularize if singular
        eps = 1e-8
        S.flat[:: S.shape[0] + 1] += eps
        VI = np.linalg.pinv(S)
        Dt = _cdist_mahalanobis(Xt, Xc, VI)
        Dc = Dt.T.copy()
        return Dt, Dc

    raise ValueError(f"Unknown distance_metric: {metric}")


def _cdist_sq(A, B):
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    a2 = np.sum(A * A, axis=1, keepdims=True)
    b2 = np.sum(B * B, axis=1, keepdims=True).T
    return a2 + b2 - 2.0 * (A @ B.T)


def _cdist_minkowski(A, B, p=2):
    # Compute pairwise Minkowski distances
    # shape: (len(A), len(B))
    # Uses broadcasting; may be memory heavy if very large. Works fine for typical matching sizes.
    diff = A[:, None, :] - B[None, :, :]
    return np.power(np.sum(np.power(np.abs(diff), p), axis=2), 1.0 / p)


def _cdist_mahalanobis(A, B, VI):
    # d(a,b) = sqrt((a-b)ᵀ VI (a-b))
    diff = A[:, None, :] - B[None, :, :]
    # (N_a, N_b, d) x (d,d) -> (N_a, N_b, d)
    tmp = diff @ VI
    d2 = np.einsum('ijk,ijk->ij', tmp, diff)
    d2 = np.clip(d2, a_min=0.0, a_max=None)
    return np.sqrt(d2)


def _impute_counterfactuals_from_neighbors(opposite_outcomes, D, k=1, with_replacement=True):
    """
    Given distances D from focal units (rows) to candidate neighbors (cols),
    returns:
      cf (np.ndarray): imputed counterfactual outcomes per focal unit (mean over k nearest).
      matched_mask (np.ndarray[bool]): True where we found at least 1 finite-distance neighbor.

    If with_replacement=False, we greedily reserve neighbors (stable but not globally optimal).
    """
    n_focal, n_cand = D.shape
    cf = np.zeros(n_focal, dtype=float)
    matched = np.zeros(n_focal, dtype=bool)

    if n_cand == 0:
        return cf, matched

    if with_replacement:
        # For each focal, take k smallest finite distances
        order = np.argsort(D, axis=1)
        for i in range(n_focal):
            # valid neighbors (finite distances)
            nn = order[i]
            nn = nn[np.isfinite(D[i, nn])]
            if nn.size == 0:
                continue
            use = nn[: min(k, nn.size)]
            cf[i] = float(np.mean(opposite_outcomes[use]))
            matched[i] = True
        return cf, matched

    # Without replacement: greedy selection to avoid reusing the same neighbors too much
    taken = np.full(n_cand, fill_value=0, dtype=int)
    max_use = 1  # each candidate can be used at most once per greedy pass

    # Process focal units in order of their nearest available distance
    focal_order = np.argsort(np.nanmin(D, axis=1))
    for i in focal_order:
        # sort candidate neighbors for this focal
        idxs = np.argsort(D[i, :])
        idxs = [j for j in idxs if np.isfinite(D[i, j])]
        chosen = []
        for j in idxs:
            if taken[j] < max_use:
                taken[j] += 1
                chosen.append(j)
                if len(chosen) == k:
                    break
        if len(chosen) > 0:
            cf[i] = float(np.mean(opposite_outcomes[chosen]))
            matched[i] = True

    return cf, matched