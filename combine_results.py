"""Merge results from several colab_experiment.py runs into final tables/figures.

Use this when you split the work across multiple notebooks (e.g. main on one,
extensions on another, or datasets split across several).  Download each run's
``ca_artifacts_<label>.zip``, unzip them all under one folder, then:

    python combine_results.py <root_dir>

It recursively finds every ``ca_results_main.csv``, ``ca_results_ext.csv`` and
``curves/all_curves.json``, concatenates them (de-duplicating on
(Dataset, Seed)), and prints the merged LaTeX rows + writes a combined boxplot
and convergence figure.  Equivalent first cell for Colab:

    !python combine_results.py /content      # after unzipping all bundles there
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

PRETTY = {"poisson":"Poisson","heat":"Heat","advection":"Advection","darcy":"Darcy",
          "wave":"Wave","darcy_multi":"Darcy-multi","heat_sensor":"Heat-sensor"}
DIN = {"poisson":1,"heat":1,"advection":1,"darcy":1,"wave":2,"darcy_multi":2,"heat_sensor":2}
MAIN_ORDER = ["poisson","heat","advection","darcy","wave","darcy_multi","heat_sensor"]
EXT_ORDER  = ["heat","wave","darcy_multi","heat_sensor"]


def _concat(pattern, root):
    files = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if {"Dataset", "Seed"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["Dataset", "Seed"], keep="first")
    print(f"  {pattern}: {len(files)} file(s) -> {len(df)} rows")
    return df


def _load_curves(root):
    out = {}
    for f in glob.glob(os.path.join(root, "**", "all_curves.json"), recursive=True):
        out.update(json.load(open(f)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--epochs", type=int, default=None, help="override epoch count for x-axis")
    args = ap.parse_args()

    print("Merging from", os.path.abspath(args.root))
    main_df = _concat("ca_results_main.csv", args.root)
    ext_df  = _concat("ca_results_ext.csv", args.root)
    curves  = _load_curves(args.root)
    main_df.to_csv("merged_main.csv", index=False)
    ext_df.to_csv("merged_ext.csv", index=False)

    def has(df, col, ds): return (not df.empty) and col in df and not df[df.Dataset==ds][col].dropna().empty

    print("\n% RESULTS_MAIN rows")
    for ds in MAIN_ORDER:
        if not (has(main_df,"std_loss",ds) and has(main_df,"ca_loss",ds)): continue
        s=main_df[main_df.Dataset==ds]; a,b=s.std_loss.values,s.ca_loss.values
        imp=(a-b)/a*100; win=np.mean(b<a)*100
        p=wilcoxon(a,b).pvalue if (wilcoxon and len(a)>1 and np.any(a!=b)) else float("nan")
        print(f"{PRETTY[ds]:<13s} & {DIN[ds]} & {a.mean():.4f} $\\pm$ {a.std():.4f} & "
              f"{b.mean():.4f} $\\pm$ {b.std():.4f} & {imp.mean():+.1f} & {win:.0f} & {p:.3f} \\\\")

    print("\n% RESULTS_TIME rows")
    for ds in ["heat","darcy_multi"]:
        if not has(main_df,"std_time",ds): continue
        s=main_df[main_df.Dataset==ds]
        pre=ext_df[ext_df.Dataset==ds].ll_precomp_time.mean() if has(ext_df,"ll_precomp_time",ds) else float("nan")
        print(f"{PRETTY[ds]:<12s} & {s.std_time.mean():.2f} & {s.ca_time.mean():.2f} & {pre:.2f} \\\\")

    print("\n% RESULTS_EXT rows")
    for ds in EXT_ORDER:
        if ext_df.empty or ext_df[ext_df.Dataset==ds].empty: continue
        s=ext_df[ext_df.Dataset==ds]
        def g(c): return f"{s[c].mean():.4f}" if (c in s and not s[c].isna().all()) else "--"
        print(f"{PRETTY[ds]:<12s} & {g('std_loss')} & {g('ca_loss')} & {g('ll_loss')} & {g('per_layer_loss')} & {g('dynamic_loss')} \\\\")

    # combined boxplot
    box=[ds for ds in MAIN_ORDER if has(main_df,"std_loss",ds) and has(main_df,"ca_loss",ds)]
    if box:
        fig,axes=plt.subplots(1,len(box),figsize=(3*len(box),4),squeeze=False)
        for ax,ds in zip(axes[0],box):
            s=main_df[main_df.Dataset==ds]
            ax.boxplot([s.std_loss.values,s.ca_loss.values],labels=["Std","CA"],patch_artist=True)
            ax.set_title(ds); ax.set_yscale("log"); ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("test rel. $L_2$")
        plt.tight_layout(); plt.savefig("benchmark_boxplot.png",dpi=130,bbox_inches="tight"); plt.close()
        print("\nwrote benchmark_boxplot.png")

    # combined convergence
    def stack(ds,model):
        ks=[k for k in curves if k.startswith(f"{ds}|{model}|")]
        return np.array([curves[k]["test_relL2"] for k in ks]) if ks else None
    cd=[ds for ds in ["heat","wave","heat_sensor"] if stack(ds,"std") is not None and stack(ds,"ca") is not None]
    if cd:
        fig,axes=plt.subplots(1,len(cd),figsize=(5*len(cd),4),squeeze=False)
        for ax,ds in zip(axes[0],cd):
            ns=stack(ds,"std").shape[1]; ep=np.arange(1,ns+1)
            for model,lab,col in [("std","Standard FNO","C0"),("ca","CA-FNO","C1")]:
                arr=stack(ds,model); m,s=arr.mean(0),arr.std(0)
                ax.plot(ep,m,col,label=lab); ax.fill_between(ep,m-s,m+s,alpha=0.2,color=col)
            ax.set_title(ds); ax.set_xlabel("epoch"); ax.set_ylabel("test rel. $L_2$")
            ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout(); plt.savefig("convergence.png",dpi=130,bbox_inches="tight"); plt.close()
        print("wrote convergence.png")
    print("\nwrote merged_main.csv, merged_ext.csv")


if __name__ == "__main__":
    main()
