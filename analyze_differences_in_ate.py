#!/usr/bin/env python3
"""
Analyze differences between ATE estimation methods using one-way repeated-measures ANOVA.

Inputs:
  - bootstraps/current.csv: rows are bootstrap resamples; columns are ATE estimates for methods

Behavior:
  - Treat each row as a subject (bootstrap replicate) and each column as a within-subject condition (method).
  - Run a within-subjects (repeated-measures) ANOVA to test if mean ATE differs across methods.
  - No post-hoc analysis is performed.

Usage:
  python analyze_differences_in_ate.py
  python analyze_differences_in_ate.py --csv bootstraps/current.csv
  python analyze_differences_in_ate.py --methods backdoor.linear_regression,backdoor.T_learner
  python analyze_differences_in_ate.py --outdir statistical_analysis --print-means --sphericity

Notes:
  - Requires pandas and scipy (scikit-learn depends on scipy, so it should be available).
  - If any selected columns contain NaNs, rows with NaNs across the selected columns are dropped.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import json
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, chi2 as chi2_dist
import matplotlib.pyplot as plt
import seaborn as sns


CSV_DEFAULT = Path("bootstraps/current.csv")
OUTDIR_DEFAULT = Path("statistical_analysis")


def parse_methods_arg(methods_arg: str | None) -> List[str] | None:
    if not methods_arg:
        return None
    # Split by comma and strip whitespace
    methods = [m.strip() for m in methods_arg.split(",") if m.strip()]
    return methods if methods else None


def rm_anova_matrix(X: np.ndarray) -> dict:
    """Compute one-way repeated-measures ANOVA given matrix X with shape (n_subjects, n_methods).

    Returns a dict with keys: n, k, ss_total, ss_subjects, ss_treatments, ss_error,
    df_subjects, df_treatments, df_error, ms_treatments, ms_error, F, p, partial_eta_sq.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array [subjects x methods].")
    n, k = X.shape
    if n < 2 or k < 2:
        raise ValueError("Need at least 2 subjects and 2 methods for RM ANOVA.")

    GM = float(X.mean())
    subj_means = X.mean(axis=1)
    meth_means = X.mean(axis=0)

    ss_total = float(((X - GM) ** 2).sum())
    ss_subjects = float(k * ((subj_means - GM) ** 2).sum())
    ss_treatments = float(n * ((meth_means - GM) ** 2).sum())
    ss_error = ss_total - ss_subjects - ss_treatments
    # Numerical guard: tiny negatives -> 0
    if ss_error < 0 and abs(ss_error) < 1e-10:
        ss_error = 0.0

    df_subjects = n - 1
    df_treatments = k - 1
    df_error = df_subjects * df_treatments
    if df_error <= 0:
        raise ValueError("Non-positive error degrees of freedom; check data shape.")

    ms_treatments = ss_treatments / df_treatments
    ms_error = ss_error / df_error
    F = ms_treatments / ms_error if ms_error > 0 else np.inf
    p = float(f_dist.sf(F, df_treatments, df_error)) if np.isfinite(F) else 0.0
    partial_eta_sq = ss_treatments / (ss_treatments + ss_error) if (ss_treatments + ss_error) > 0 else np.nan

    return {
        "n": n,
        "k": k,
        "ss_total": ss_total,
        "ss_subjects": ss_subjects,
        "ss_treatments": ss_treatments,
        "ss_error": ss_error,
        "df_subjects": df_subjects,
        "df_treatments": df_treatments,
        "df_error": df_error,
        "ms_treatments": ms_treatments,
        "ms_error": ms_error,
        "F": F,
        "p": p,
        "partial_eta_sq": partial_eta_sq,
    }

def mauchly_sphericity_test(X: np.ndarray) -> dict:
    """Mauchly's test of sphericity for one-way repeated measures.

    Returns dict with: p, W, chi2, df, n, k. Requires k>=3.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array [subjects x methods].")
    n, k = X.shape
    if k < 3:
        raise ValueError("Sphericity test undefined for fewer than 3 levels.")

    # Sample covariance across methods (columns), ddof=1
    S = np.cov(X, rowvar=False, ddof=1)
    # Numerical stability: ensure positive-definite for log-det
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        # add tiny ridge and retry
        ridge = 1e-12 * np.trace(S) / k if np.isfinite(np.trace(S)) and np.trace(S) > 0 else 1e-12
        S = S + np.eye(k) * ridge
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            raise np.linalg.LinAlgError("Covariance matrix not positive definite for Mauchly's test.")

    traceS = float(np.trace(S))
    if not np.isfinite(traceS) or traceS <= 0:
        raise ValueError("Invalid covariance trace for Mauchly's test.")

    logW = logdet - k * (np.log(traceS) - np.log(k))
    W = float(np.exp(logW))

    c = (2 * k * k + k + 2) / (6 * (k - 1) * (n - 1))
    df = int(k * (k - 1) / 2 - 1)
    if df <= 0:
        raise ValueError("Non-positive degrees of freedom for Mauchly's test.")
    chi2 = float(-(n - 1) * (1 - c) * np.log(W))
    p = float(chi2_dist.sf(chi2, df))

    return {"p": p, "W": W, "chi2": chi2, "df": df, "n": n, "k": k}


def main():
    ap = argparse.ArgumentParser(description="RM ANOVA across ATE estimation methods (within-subjects: bootstrap replicates).")
    ap.add_argument("--csv", type=Path, default=CSV_DEFAULT, help="Path to bootstraps/current.csv")
    ap.add_argument(
        "--methods",
        help="Comma-separated list of column names to include (default: all numeric columns).",
    )
    ap.add_argument("--print-means", action="store_true", help="Print per-method mean ATE before ANOVA results.")
    ap.add_argument("--outdir", type=Path, default=OUTDIR_DEFAULT, help="Directory to save analysis outputs.")
    ap.add_argument("--sphericity", action="store_true", help="Run and report Mauchly's sphericity test.")
    ap.add_argument("--no-plot", action="store_true", help="Do not generate the per-method bar plot.")
    ap.add_argument("--no-pairwise", action="store_true", help="Do not generate pairwise mean-difference table.")
    args = ap.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    # Select methods
    selected_methods = parse_methods_arg(args.methods)
    if selected_methods is None:
        # Default to all numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError("No numeric columns found in CSV; provide --methods explicitly.")
        methods = num_cols
    else:
        missing = [m for m in selected_methods if m not in df.columns]
        if missing:
            raise KeyError(f"Methods not found in CSV columns: {missing}")
        methods = selected_methods

    data = df[methods].copy()
    # Drop rows with NaNs across the selected methods
    n_before = len(data)
    data = data.dropna(axis=0, how="any")
    n_after = len(data)
    if n_after < n_before:
        print(f"[info] Dropped {n_before - n_after} rows with NaNs across selected methods.")

    if data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError("Insufficient data after NaN-drop: need ≥2 rows and ≥2 methods.")

    X = data.to_numpy(dtype=float)
    res = rm_anova_matrix(X)

    # Output
    print("=== RM ANOVA (Within-Subjects: Bootstrap Replicates) ===")
    print(f"CSV: {args.csv}")
    print(f"Subjects (bootstraps): {res['n']}")
    print(f"Methods (conditions):   {res['k']}")
    print("Methods:")
    for m in methods:
        print(f"  - {m}")

    meth_means = data.mean(axis=0)
    if args.print_means:
        print("\nPer-method mean ATE:")
        for m in methods:
            print(f"  {m}: {meth_means[m]:.6f}")

    print("\nANOVA Table (one-way RM)")
    print(f"  Between-methods: SS={res['ss_treatments']:.6f}, df={res['df_treatments']}, MS={res['ms_treatments']:.6f}")
    print(f"  Error (residual): SS={res['ss_error']:.6f}, df={res['df_error']}, MS={res['ms_error']:.6f}")
    print(f"  Subjects:         SS={res['ss_subjects']:.6f}, df={res['df_subjects']}")
    print(f"  Total:            SS={res['ss_total']:.6f}")

    print("\nTest:")
    print(f"  F({res['df_treatments']}, {res['df_error']}) = {res['F']:.6f}")
    print(f"  p-value = {res['p']:.6e}")
    if np.isfinite(res["partial_eta_sq"]):
        print(f"  partial eta^2 = {res['partial_eta_sq']:.6f}")
    else:
        print("  partial eta^2 = NaN")

    # Optional sphericity test (Mauchly)
    sph = None
    if args.sphericity:
        if data.shape[1] < 3:
            print("\nSphericity: skipped (needs ≥3 levels).")
        else:
            try:
                sph = mauchly_sphericity_test(X)
                print("\nMauchly's Test of Sphericity:")
                print(f"  W = {sph['W']:.6f}")
                print(f"  chi2({sph['df']}) = {sph['chi2']:.6f}")
                print(f"  p-value = {sph['p']:.6e}")
            except Exception as e:
                print(f"\nSphericity: error computing Mauchly's test: {e}")

    print("\nNotes:")
    print("  - No post-hoc analysis performed.")
    if args.sphericity:
        print("  - Sphericity tested via Mauchly's test (approximate chi-square).")
    else:
        print("  - Sphericity not tested; interpret p-value with standard assumptions.")

    # Save outputs
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Means
    means_path = outdir / "per_method_means.csv"
    meth_means.to_csv(means_path, header=["mean_ate"])  # index=method names

    # 2) ANOVA summary JSON
    anova_json = {
        "csv": str(args.csv),
        "methods": methods,
        "n_subjects": int(res["n"]),
        "n_methods": int(res["k"]),
        "ss_total": res["ss_total"],
        "ss_subjects": res["ss_subjects"],
        "ss_treatments": res["ss_treatments"],
        "ss_error": res["ss_error"],
        "df_subjects": int(res["df_subjects"]),
        "df_treatments": int(res["df_treatments"]),
        "df_error": int(res["df_error"]),
        "ms_treatments": res["ms_treatments"],
        "ms_error": res["ms_error"],
        "F": res["F"],
        "p": res["p"],
        "partial_eta_sq": res["partial_eta_sq"],
        "mauchly": sph,
    }
    (outdir / "rm_anova_results.json").write_text(json.dumps(anova_json, indent=2))

    # 3) ANOVA table CSV-like
    table_rows = [
        {"source": "Methods", "SS": res["ss_treatments"], "df": int(res["df_treatments"]), "MS": res["ms_treatments"], "F": res["F"], "p": res["p"]},
        {"source": "Error",   "SS": res["ss_error"],      "df": int(res["df_error"]),      "MS": res["ms_error"]},
        {"source": "Subjects","SS": res["ss_subjects"],   "df": int(res["df_subjects"])},
        {"source": "Total",   "SS": res["ss_total"]},
    ]
    pd.DataFrame(table_rows).to_csv(outdir / "rm_anova_table.csv", index=False)

    # 4) Bar plot with 95% CI (percentile across bootstraps)
    if not args.no_plot:
        # Compute percentile CIs per method
        ci_lo = data.quantile(0.025, axis=0)
        ci_hi = data.quantile(0.975, axis=0)
        # Short labels for nicer plotting
        label_map = {
            "backdoor.propensity_score_matching": "PS Matching",
            "backdoor.propensity_score_weighting": "PS Weighting",
            "backdoor.propensity_score_stratification": "PS Stratification",
            "backdoor.linear_regression": "Linear Regression",
            "backdoor.distance_matching": "Distance Matching",
            "backdoor.T_learner": "T-learner",
        }
        def short_label(name: str) -> str:
            if name in label_map:
                return label_map[name]
            s = name.replace("backdoor.", "").replace("_", " ")
            return s

        summary = pd.DataFrame({
            "method": data.columns,
            "method_short": [short_label(c) for c in data.columns],
            "mean": meth_means.values,
            "ci_lo": ci_lo.values,
            "ci_hi": ci_hi.values,
        })
        # Sort by mean desc for nicer presentation
        summary = summary.sort_values("mean", ascending=False).reset_index(drop=True)

        plt.figure(figsize=(8, 5), dpi=160)
        sns.set_style("whitegrid")
        x = np.arange(len(summary))
        y = summary["mean"].to_numpy()
        yerr = np.vstack([y - summary["ci_lo"].to_numpy(), summary["ci_hi"].to_numpy() - y])
        plt.bar(x, y, color="#4C78A8")
        plt.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#333333", capsize=4, linewidth=1.2)
        plt.xticks(x, summary["method_short"], rotation=20, ha="right")
        plt.ylabel("ATE (mean across bootstraps)")
        plt.title("ATE by Method with 95% Bootstrap CI")
        plt.tight_layout()
        fig_base = outdir / "per_method_means_barplot"
        plt.savefig(f"{fig_base}.png", bbox_inches="tight")
        plt.savefig(f"{fig_base}.svg", bbox_inches="tight")
        plt.close()

    # 5) Pairwise mean differences with 95% CI (descriptive)
    if not args.no_pairwise:
        rows = []
        label_map = {
            "backdoor.propensity_score_matching": "PS Matching",
            "backdoor.propensity_score_weighting": "PS Weighting",
            "backdoor.propensity_score_stratification": "PS Stratification",
            "backdoor.linear_regression": "Linear Regression",
            "backdoor.distance_matching": "Distance Matching",
            "backdoor.T_learner": "T-learner",
        }
        def short_label(name: str) -> str:
            if name in label_map:
                return label_map[name]
            s = name.replace("backdoor.", "").replace("_", " ")
            return s

        cols = list(data.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                diffs = (data[a] - data[b]).to_numpy()
                rows.append({
                    "method_a": a,
                    "method_b": b,
                    "method_a_short": short_label(a),
                    "method_b_short": short_label(b),
                    "mean_diff": float(np.mean(diffs)),
                    "ci2.5": float(np.percentile(diffs, 2.5)),
                    "ci97.5": float(np.percentile(diffs, 97.5)),
                    "sd": float(np.std(diffs, ddof=1)),
                })
        pw_df = pd.DataFrame(rows)
        pw_df.to_csv(outdir / "pairwise_mean_differences.csv", index=False)

        # LaTeX table for pairwise differences (descriptive)
        def tex_escape(s: str) -> str:
            return s.replace("_", "\\_")

        # Round for readability
        pw_disp = pw_df.copy()
        for col in ("mean_diff", "ci2.5", "ci97.5", "sd"):
            pw_disp[col] = pw_disp[col].map(lambda v: f"{v:.3f}")

        lines = []
        lines.append("% Auto-generated by analyze_differences_in_ate.py")
        lines.append("% Mean difference defined as method_a - method_b")
        lines.append(r"\begin{table*}[htbp]")
        lines.append(r"  \centering")
        lines.append(r"  \caption{Descriptive pairwise mean differences in ATE across bootstraps with 95\% bootstrap CIs.}")
        lines.append(r"  \label{tab:pairwise-ate-diffs}")
        lines.append(r"  \begin{tabular}{llrrrr}")
        lines.append(r"    \hline")
        lines.append(r"    Method A & Method B & Mean diff & 2.5\% & 97.5\% & SD \\")
        lines.append(r"    \hline")
        for _, r in pw_disp.iterrows():
            a = tex_escape(r["method_a_short"]) if isinstance(r["method_a_short"], str) else tex_escape(r["method_a"]) 
            b = tex_escape(r["method_b_short"]) if isinstance(r["method_b_short"], str) else tex_escape(r["method_b"]) 
            lines.append(
                f"    {a} & {b} & {r['mean_diff']} & {r['ci2.5']} & {r['ci97.5']} & {r['sd']} \\")
        lines.append(r"    \hline")
        lines.append(r"  \end{tabular}")
        lines.append(r"  \vspace{0.5em}")
        lines.append(r"  {\footnotesize Note: No multiple-comparison correction applied; values are descriptive. Mean difference is Method A minus Method B.}")
        lines.append(r"\end{table*}")

        (outdir / "pairwise_mean_differences.tex").write_text("\n".join(lines))

    print(f"\nSaved outputs to: {outdir}")


if __name__ == "__main__":
    main()
