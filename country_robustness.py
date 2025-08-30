#!/usr/bin/env python3
"""
Plot Europe (continental) with:
- Color = mean of method scores per country
- Alpha = variance of method scores per country
Options:
- Percentile clipping for mean (avoid one extreme country washing others)
- Flip encodings (color<->alpha)
- Country labels placed inside polygons

Inputs:
  data/res_per_country.json
  data/dictionary.json  (expects ['Y11_Country']['values'] mapping: "1":"Austria",...)

Output:
  out/europe_mean_var.svg

Usage examples:
  python plot_europe_mean_var.py
  python plot_europe_mean_var.py --mean-clip-upper 95
  python plot_europe_mean_var.py --flip
  python plot_europe_mean_var.py --alpha-min 0.3 --alpha-max 0.95 --label-size 7
"""

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box

# ----------------------------
# Defaults / config
# ----------------------------
RES_PATH_DEFAULT = Path("data/results_per_country.json")
DICT_PATH_DEFAULT = Path("data/dictionary.json")
OUT_PATH_DEFAULT  = Path("figs/europe_mean_var.svg")
OUT_PATH_DEFAULT.parent.mkdir(parents=True, exist_ok=True)

# Natural Earth 110m admin_0 countries
NE_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"

# Bounding box for continental Europe (lon_min, lat_min, lon_max, lat_max)
EU_BBOX_DEFAULT = (-25.0, 34.0, 45.0, 72.0)

# Manual name → ISO-3 map (based on your list)
NAME_TO_ISO3 = {
    "Austria": "AUT",
    "Belgium": "BEL",
    "Bulgaria": "BGR",
    "Cyprus": "CYP",
    "Czech Republic": "CZE",
    "Germany": "DEU",
    "Denmark": "DNK",
    "Estonia": "EST",
    "Greece": "GRC",
    "Spain": "ESP",
    "Finland": "FIN",
    "France": "FRA",
    "Hungary": "HUN",
    "Ireland": "IRL",
    "Italy": "ITA",
    "Lithuania": "LTU",
    "Luxembourg": "LUX",
    "Latvia": "LVA",
    "Malta": "MLT",
    "Netherlands": "NLD",
    "Poland": "POL",
    "Portugal": "PRT",
    "Romania": "ROU",
    "Sweden": "SWE",
    "Slovenia": "SVN",
    "Slovakia": "SVK",
    "United Kingdom": "GBR",
    "Turkey": "TUR",
    "Croatia": "HRV",
    "North Macedonia": "MKD",
    "Norway": "NOR",
}

def warn(*msg):
    print("[warn]", *msg, file=sys.stderr)

def load_world():
    world = gpd.read_file(NE_URL)
    # Normalize ISO3 col
    if "iso_a3" not in world.columns:
        for cand in ("ADM0_A3", "ISO_A3", "ADM0_A3_US"):
            if cand in world.columns:
                world = world.rename(columns={cand: "iso_a3"})
                break
    if "iso_a3" not in world.columns:
        raise KeyError("Could not find ISO3 code column in Natural Earth file.")
    # Normalize continent col
    if "continent" not in world.columns and "CONTINENT" in world.columns:
        world = world.rename(columns={"CONTINENT": "continent"})
    return world

def compute_country_stats(res_by_id: dict, id_to_name: dict) -> pd.DataFrame:
    rows = []
    for cid_str, methods_dict in res_by_id.items():
        cname = id_to_name.get(cid_str)
        if not cname:
            warn(f"id {cid_str} missing in dictionary.json")
            continue
        iso = NAME_TO_ISO3.get(cname)
        if not iso:
            warn(f"No ISO3 mapping for '{cname}' (id={cid_str})")
            continue
        vals = [v for v in methods_dict.values() if isinstance(v, (int, float))]
        if not vals:
            warn(f"No numeric method values for '{cname}' (id={cid_str})")
            continue
        rows.append({
            "iso_a3": iso,
            "mean": float(np.mean(vals)),
            "var":  float(np.var(vals, ddof=0)),
            "n_methods": len(vals),
            "name_src": cname
        })
    return pd.DataFrame(rows)

def minmax_scale(series: pd.Series):
    s = series.astype(float)
    vmin, vmax = s.min(), s.max()
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - vmin) / (vmax - vmin)

def parse_bbox(bbox_str: str):
    parts = [float(x) for x in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must be 'minx,miny,maxx,maxy'")
    return tuple(parts)

def main():
    ap = argparse.ArgumentParser(description="Europe mean/variance choropleth (color=mean, alpha=var).")
    ap.add_argument("--res", type=Path, default=RES_PATH_DEFAULT, help="Path to data/res_per_country.json")
    ap.add_argument("--dict", type=Path, default=DICT_PATH_DEFAULT, help="Path to data/dictionary.json")
    ap.add_argument("--out",  type=Path, default=OUT_PATH_DEFAULT, help="Output image path (.svg/.png)")
    ap.add_argument("--cmap", default="YlOrRd", help="Matplotlib colormap for color metric")
    ap.add_argument("--alpha-min", type=float, default=0.45, help="Minimum fill alpha")
    ap.add_argument("--alpha-max", type=float, default=1.00, help="Maximum fill alpha")
    ap.add_argument("--mean-clip-lower", type=float, default=0.0, help="Lower percentile to clip mean (0-100)")
    ap.add_argument("--mean-clip-upper", type=float, default=100.0, help="Upper percentile to clip mean (0-100)")
    ap.add_argument("--flip", action="store_true", help="Flip encodings: color=variance, alpha=mean")
    ap.add_argument("--bbox", default="{}, {}, {}, {}".format(*EU_BBOX_DEFAULT),
                    help="BBox 'minx,miny,maxx,maxy' for continental Europe clipping")
    ap.add_argument("--label-size", type=float, default=8.0, help="Country label font size")
    ap.add_argument("--labels-all", action="store_true",
                    help="Label all visible countries (default: only those with data)")
    args = ap.parse_args()

    # ------- load inputs -------
    with open(args.res, "r", encoding="utf-8") as f:
        res_by_id = json.load(f)
    with open(args.dict, "r", encoding="utf-8") as f:
        dct = json.load(f)
    try:
        id_to_name = dict(dct["Y11_Country"]["values"])
    except Exception as e:
        raise KeyError("Expected ['Y11_Country']['values'] mapping in dictionary.json") from e

    stats = compute_country_stats(res_by_id, id_to_name)
    if stats.empty:
        raise RuntimeError("No stats computed; check inputs and name→ISO mapping.")

    # ------- load & prep map -------
    world = load_world()
    if "continent" in world.columns:
        eu = world[world["continent"].str.lower().eq("europe")].copy()
    else:
        eu = world.copy()

    # Clip to bbox
    minx, miny, maxx, maxy = parse_bbox(args.bbox)
    bbox_geom = box(minx, miny, maxx, maxy)
    eu["geometry"] = eu["geometry"].intersection(bbox_geom)
    eu = eu[eu.geometry.notna() & ~eu.is_empty].copy()

    # Join data
    eu = eu.merge(stats, on="iso_a3", how="left")

    # Determine which field drives color vs alpha
    color_field = "var" if args.flip else "mean"
    alpha_field = "mean" if args.flip else "var"

    # -------- percentile clip on mean (to avoid outlier washout) --------
    # We only clip the 'mean' values before normalization; variance stays untouched.
    mean_vals = eu["mean"].copy()
    if mean_vals.notna().any():
        lo = np.nanpercentile(mean_vals, args.mean_clip_lower)
        hi = np.nanpercentile(mean_vals, args.mean_clip_upper)
        eu["mean_clipped"] = np.clip(mean_vals, lo, hi)
    else:
        eu["mean_clipped"] = mean_vals

    # Normalized metrics
    # Color normalization uses 'color_field' (mean_clipped if color_field==mean)
    if color_field == "mean":
        col_source = eu["mean_clipped"]
        col_label  = "Mean ATE"
    else:
        col_source = eu[color_field]
        col_label  = "Variance"

    alp_source = eu["mean_clipped"] if alpha_field == "mean" else eu[alpha_field]

    eu["color_norm"] = minmax_scale(col_source)
    eu["alpha_norm"] = minmax_scale(alp_source)

    # Map alpha into [alpha_min, alpha_max]
    a0, a1 = float(args.alpha_min), float(args.alpha_max)
    eu["alpha_val"] = a0 + eu["alpha_norm"] * (a1 - a0)

    # Build RGBA
    cmap = plt.get_cmap(args.cmap)
    def rgba(row):
        cn = row["color_norm"]
        if pd.notna(cn):
            r, g, b, _ = cmap(cn)
            return (r, g, b, float(row["alpha_val"]))
        # missing -> light grey, faint
        r, g, b = mcolors.to_rgb("#999999")
        return (r, g, b, 0.45)

    eu["rgba"] = eu.apply(rgba, axis=1)

    # -------- plot --------
    fig, ax = plt.subplots(figsize=(10, 8), dpi=220)
    eu.plot(color=eu["rgba"], edgecolor="white", linewidth=0.5, ax=ax)
    ax.set_axis_off()

    # Colorbar reflects the color metric scale
    if col_source.notna().any():
        vmin, vmax = float(np.nanmin(col_source)), float(np.nanmax(col_source))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
        cbar.set_label(col_label)

    # Alpha explainer
    alpha_label = "Mean (clipped)" if alpha_field == "mean" else "Variance"
    ax.text(0.01, 0.02,
            f"Opacity ∝ {alpha_label}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10)

    # -------- labels (inside each polygon) --------
    # Use representative_point() to ensure the label falls within the shape
    # Label only countries with data unless --labels-all is set
    label_gdf = eu.copy()
    if not args.labels_all:
        label_gdf = label_gdf[label_gdf["mean"].notna() | label_gdf["var"].notna()]

    # Determine a name column
    name_col = None
    for cand in ("NAME_EN", "NAME", "name", "ADMIN", "SOVEREIGNT"):
        if cand in label_gdf.columns:
            name_col = cand
            break
    if name_col is None:
        name_col = "iso_a3"

    reps = label_gdf.representative_point()
    for (x, y), name in zip(reps.geometry.apply(lambda g: (g.x, g.y)), label_gdf[name_col]):
        ax.text(
            x, y, str(name),
            fontsize=args.label_size,
            ha="center", va="center",
            color="black",
            path_effects=[pe.withStroke(linewidth=1.7, foreground="white", alpha=0.9)]
        )

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    plt.show()
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
