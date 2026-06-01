"""Turn benchmark_raw.csv into LaTeX table rows for the writeup.

Usage:  python make_report.py <raw_csv> [--out report.txt]

Emits the three result tables (main std-vs-ca, timing, extensions) as LaTeX
rows, plus paired Wilcoxon signed-rank p-values.  Paste the rows between the
RESULTS_*_BEGIN/END markers in writeup/main.tex.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PRETTY = {
    "poisson": "Poisson", "heat": "Heat", "advection": "Advection",
    "darcy": "Darcy", "wave": "Wave", "darcy_multi": "Darcy-multi",
    "heat_sensor": "Heat-sensor",
}
DIN = {"poisson": 1, "heat": 1, "advection": 1, "darcy": 1,
       "wave": 2, "darcy_multi": 2, "heat_sensor": 2}
MAIN_ORDER = ["poisson", "heat", "advection", "darcy", "wave", "darcy_multi", "heat_sensor"]
EXT_ORDER = ["heat", "wave", "darcy_multi", "heat_sensor"]


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    lines = []

    lines.append("% ===== MAIN TABLE (std vs ca) =====")
    for ds in MAIN_ORDER:
        sub = df[df["Dataset"] == ds]
        if sub.empty or "ca_loss" not in sub or sub["ca_loss"].dropna().empty:
            continue
        s = sub["std_loss"].values
        c = sub["ca_loss"].values
        mask = ~(np.isnan(s) | np.isnan(c))
        s, c = s[mask], c[mask]
        imp = (s - c) / s * 100.0
        win = float(np.mean(c < s) * 100.0)
        try:
            p = wilcoxon(s, c).pvalue
        except ValueError:
            p = float("nan")
        lines.append(
            f"{PRETTY[ds]:<15s} & {DIN[ds]} & {fmt(s.mean())} $\\pm$ {fmt(s.std())} "
            f"& {fmt(c.mean())} $\\pm$ {fmt(c.std())} & {imp.mean():+.1f} & {win:.0f} "
            f"& {p:.3f} \\\\"
        )

    lines.append("\n% ===== TIMING TABLE =====")
    for ds in ["heat", "darcy_multi"]:
        sub = df[df["Dataset"] == ds]
        if sub.empty:
            continue
        st = sub["std_time"].mean() if "std_time" in sub else float("nan")
        ct = sub["ca_time"].mean() if "ca_time" in sub else float("nan")
        pt = sub["ll_precomp_time"].mean() if "ll_precomp_time" in sub else float("nan")
        lines.append(f"{PRETTY[ds]:<12s} & {st:.2f} & {ct:.2f} & {pt:.2f} \\\\")

    lines.append("\n% ===== EXTENSIONS TABLE =====")
    cols = [("std_loss", "std"), ("ca_loss", "ca"), ("ll_loss", "ll"),
            ("per_layer_loss", "pl"), ("dynamic_loss", "dyn")]
    for ds in EXT_ORDER:
        sub = df[df["Dataset"] == ds]
        if sub.empty:
            continue
        vals = []
        for col, _ in cols:
            v = sub[col].dropna().mean() if col in sub else float("nan")
            vals.append("--" if np.isnan(v) else fmt(v))
        lines.append(f"{PRETTY[ds]:<12s} & " + " & ".join(vals) + " \\\\")

    out = "\n".join(lines)
    print(out)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(out + "\n")


if __name__ == "__main__":
    main()
