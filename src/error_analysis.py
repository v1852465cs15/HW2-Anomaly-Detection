"""Group C: false-positive / false-negative analysis of the Tier 2 autoencoder (paper section 8).

  python "part c/src/error_analysis.py"

Uses Group B's final run (t2_ae__feat-all__cont-0__score-ce__tau-f1), seed 0, test split:
the model of the main results table and of Group B's figures (tau = 6.70 nats).
Analysis only: nothing here is used to fit or tune any model.

Writes
  part c/results/error_all_cases.csv      every FP / FN with its diagnostics
  part c/results/error_analysis.md        breakdown tables + selected case studies
  part c/figures/fig_error_analysis.png   3-panel figure

Diagnostics per test site
  ratio        s(x) / tau. Near miss: within a factor 2 of tau. Inversion: off by a factor >= 10.
  h_legit      Hamming distance (number of differing features) to the closest LEGITIMATE training site
  h_phish      same, to the closest PHISHING training site
  conflict     the same 30-feature vector carries both labels somewhere in the dataset (label noise)
  seeds_wrong  in how many of the 5 seeds this site is misclassified (5 = systematic blind spot)
  xgb_pred     what the supervised Tier 3 model (seed 0) predicts for the same site
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HW2 = Path(__file__).resolve().parents[2]
PB = HW2 / "part b"
OUT, FIG = HW2 / "part c" / "results", HW2 / "part c" / "figures"
RUNS = PB / "results" / "runs"
AE_RUN = RUNS / "t2_ae__feat-all__cont-0__score-ce__tau-f1"
XGB_RUN = RUNS / "t3_xgb__feat-all__cont-0__score-ce__tau-f1"
SEED = 0


def load():
    tr, va, te = (pd.read_csv(PB / "data" / f"{s}.csv") for s in ("train", "val", "test"))
    feats = [c for c in tr.columns if c not in ("row_id", "Result")]
    return tr, va, te, feats


def outcome(pred, y):
    return np.select([(pred == 1) & (y == 1), (pred == 1) & (y == 0), (pred == 0) & (y == 1)],
                     ["TP", "FP", "FN"], "TN")


def build_table(tr, va, te, feats):
    p = pd.read_csv(AE_RUN / "predictions.csv")
    p = p[p["split"] == "test"]
    p0 = p[p["seed"] == SEED].set_index("row_id")
    df = pd.DataFrame({"score": p0["score"], "tau": p0["tau"], "y": p0["is_phishing"],
                       "pred": p0["pred"]})
    df["outcome"] = outcome(df["pred"].to_numpy(), df["y"].to_numpy())
    df["ratio"] = df["score"] / df["tau"]
    lr = np.abs(np.log10(df["ratio"]))
    df["distance_to_tau"] = np.select([lr < np.log10(2), lr >= 1], ["near miss (<2x)", "inversion (>=10x)"],
                                      "moderate (2-10x)")

    # nearest training neighbours (Hamming distance on the 30 codes)
    Xte = te.set_index("row_id").loc[df.index, feats].to_numpy()
    Xtr, ytr = tr[feats].to_numpy(), tr["Result"].to_numpy()
    D = (Xte[:, None, :] != Xtr[None, :, :]).sum(axis=2)
    df["h_legit"], df["h_phish"] = D[:, ytr == 1].min(axis=1), D[:, ytr == -1].min(axis=1)
    df["n_train_copies_legit"] = (D[:, ytr == 1] == 0).sum(axis=1)
    df["n_train_copies_phish"] = (D[:, ytr == -1] == 0).sum(axis=1)

    # label conflicts anywhere in the dataset
    def key(d):
        return d[feats].astype(str).agg("".join, axis=1)
    allx = pd.concat([tr, va, te])
    n_labels = allx.assign(k=key(allx)).groupby("k")["Result"].nunique()
    df["conflict"] = key(te.set_index("row_id").loc[df.index]).map(n_labels).to_numpy() > 1

    # stability over seeds
    piv = p.pivot_table(index="row_id", columns="seed", values="pred")
    df["seeds_wrong"] = piv.ne(p.groupby("row_id")["is_phishing"].first(), axis=0).sum(axis=1)

    x = pd.read_csv(XGB_RUN / "predictions.csv")
    x0 = x[(x["split"] == "test") & (x["seed"] == SEED)].set_index("row_id")
    df["xgb_score"], df["xgb_pred"] = x0["score"], x0["pred"]
    return df


def per_feature(feats):
    pfe = pd.read_csv(AE_RUN / "per_feature_error.csv")
    return pfe[pfe["seed"] == SEED].set_index("row_id")[feats]


# ------------------------------------------------------------------ tables

def breakdown(df):
    rows = []
    for o in ("FN", "FP"):
        e = df[df["outcome"] == o]
        n = len(e)

        def pc(mask):
            return f"{int(mask.sum())} ({mask.mean():.0%})"
        rows.append({
            "error": f"{o} (n={n})",
            "near miss (<2x τ)": pc(e["distance_to_tau"] == "near miss (<2x)"),
            "inversion (>=10x)": pc(e["distance_to_tau"] == "inversion (>=10x)"),
            "median s/τ": f"{e['ratio'].median():.2f}",
            "identical to a legit train site": pc(e["h_legit"] == 0),
            "<=1 feature from a legit train site": pc(e["h_legit"] <= 1),
            "label conflict": pc(e["conflict"]),
            "wrong in all 5 seeds": pc(e["seeds_wrong"] == 5),
            "XGBoost gets it right": pc(e["xgb_pred"] != e["pred"]),
        })
    return pd.DataFrame(rows).set_index("error").T


def feature_profile(df, te, tr, feats, top=8):
    X = te.set_index("row_id").loc[df.index, feats]
    prof = pd.DataFrame({"legit train": (tr.loc[tr["Result"] == 1, feats] == -1).mean()})
    for o in ("TP", "FN", "FP", "TN"):
        prof[o] = (X[df["outcome"] == o] == -1).mean()
    prof["TP − FN"] = prof["TP"] - prof["FN"]
    prof["FP − TN"] = prof["FP"] - prof["TN"]
    return prof


def select_cases(df):
    """Deterministic picks; each illustrates one failure mode."""
    fn, fp = df[df["outcome"] == "FN"], df[df["outcome"] == "FP"]
    picks = []
    # 1 FN, inversion + label noise: lowest score among FNs that are identical to train sites of both labels
    c = fn[fn["conflict"] & (fn["h_legit"] == 0)].sort_values("score")
    picks.append((c.index[0], "FN · inversion · label conflict"))
    # 2 FN, inversion on a vector never seen in training: lowest-scoring systematic FN with h_legit >= 1
    c = fn[(fn["h_legit"] >= 1) & ~fn["conflict"] & (fn["seeds_wrong"] == 5)].sort_values("score")
    picks.append((c.index[0], "FN · inversion · looks legitimate"))
    # 3 FN, near miss: highest-scoring FN
    picks.append((fn["score"].idxmax(), "FN · near miss"))
    # 4 FP, near miss: lowest-scoring FP
    picks.append((fp["score"].idxmin(), "FP · near miss"))
    # 5 FP, strongest false alarm: highest-scoring FP
    picks.append((fp["score"].idxmax(), "FP · confident false alarm"))
    # 6 FP, flagged although the identical vector is legitimate in training (all 5 seeds)
    c = fp[(fp["h_legit"] == 0) & (fp["seeds_wrong"] == 5)].sort_values("score", ascending=False)
    if len(c):
        picks.append((c.index[0], "FP · seen-legit vector still flagged"))
    return picks


def case_table(df, picks, pfe, te, feats):
    raw = te.set_index("row_id")
    lines = ["| # | Type | row_id | s(x) | s/τ | top per-feature terms s_j (code) | h_legit / h_phish "
             "| train copies legit/phish | seeds wrong | XGB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    detail = []
    for i, (rid, kind) in enumerate(picks, 1):
        r = df.loc[rid]
        terms = pfe.loc[rid].sort_values(ascending=False)
        share = terms / terms.sum()
        top = ", ".join(f"{f}={raw.loc[rid, f]:+d}: {terms[f]:.3g} ({share[f]:.0%})" for f in terms.index[:3])
        lines.append(f"| {i} | {kind} | {rid} | {r['score']:.4g} | {r['ratio']:.3g} | {top} "
                     f"| {r['h_legit']} / {r['h_phish']} | {r['n_train_copies_legit']} / {r['n_train_copies_phish']} "
                     f"| {r['seeds_wrong']}/5 | {'phish' if r['xgb_pred'] else 'legit'} ({r['xgb_score']:.2f}) |")
        phishy = [f for f in feats if raw.loc[rid, f] == -1]
        detail.append(f"- **#{i} {rid}** ({kind}): features coded −1 ({len(phishy)}): {', '.join(phishy) or 'none'}")
    return "\n".join(lines), "\n".join(detail)


# ------------------------------------------------------------------ figure

BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, GRID = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                "#0b0b0b", "#52514e", "#e4e3df")
COLORS = {"TP": BLUE, "FN": ORANGE, "FP": AQUA, "TN": "#9a9993"}


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure(df, prof):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 9, "axes.labelsize": 8,
                         "axes.labelcolor": INK})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), gridspec_kw={"width_ratios": [1.1, 1, 1.2]})

    # (a) how far errors are from tau
    ax = axes[0]
    bins = np.logspace(-5, 1, 31)
    for o, lab in (("FN", "FN, missed"), ("FP", "FP, false alarm")):
        r = df.loc[df["outcome"] == o, "ratio"]
        ax.hist(r, bins=bins, color=COLORS[o], alpha=0.85, label=f"{lab}, n={len(r)}",
                edgecolor="white", linewidth=0.6)
    ax.set_xscale("log")
    ax.axvline(1, color=INK, ls="--", lw=1)
    ax.axvspan(0.5, 2, color=GRID, alpha=0.6, lw=0)
    top = max(np.histogram(df.loc[df["outcome"] == o, "ratio"], bins=bins)[0].max() for o in ("FN", "FP"))
    ax.set_ylim(0, top * 1.6)
    ax.text(2.3, top * 1.08, "τ\n(shaded:\nnear miss)", fontsize=7, color=MUTED, va="bottom")
    ax.set_xlabel("score relative to threshold, s(x)/τ (log scale)")
    ax.set_ylabel("test sites (seed 0)")
    ax.set_title("(a) Distance of errors to τ", loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    style(ax)

    # (b) Hamming distance to the closest legitimate training site
    ax = axes[1]
    ks = np.arange(0, 6)
    w = 0.2
    for j, o in enumerate(("TN", "FP", "FN", "TP")):
        sub = df[df["outcome"] == o]["h_legit"].clip(upper=5)
        share = [(sub == k).mean() for k in ks]
        ax.bar(ks + (j - 1.5) * (w + 0.01), share, w, color=COLORS[o], label=f"{o} (n={len(sub)})")
    ax.set_xticks(ks, ["0\n(identical)", "1", "2", "3", "4", "≥5"])
    ax.set_xlabel("features differing from closest legit training site")
    ax.set_ylabel("share of sites in group")
    ax.set_ylim(0, 0.8)
    ax.set_title("(b) Nearest legitimate neighbour", loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=1, loc="upper right")
    style(ax)

    # (c) phishing indicators that separate caught (TP) from missed (FN) phishing
    ax = axes[2]
    top = prof.sort_values("TP − FN", ascending=False).head(6).index[::-1]
    yi = np.arange(len(top))
    for off, col, lab, c, mk in ((0.18, "TP", "caught phishing (TP)", BLUE, "o"),
                                 (0.0, "FN", "missed phishing (FN)", ORANGE, "s"),
                                 (-0.18, "legit train", "legit training sites", "#9a9993", "D")):
        ax.scatter(prof.loc[top, col], yi + off, color=c, marker=mk, s=28, label=lab, zorder=3)
    for k, f in enumerate(top):
        ax.plot([prof.loc[f, "FN"], prof.loc[f, "TP"]], [k, k + 0.18], color=GRID, lw=1.5, zorder=1)
    ax.set_yticks(yi, top)
    ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel("share of sites coded −1 (phishing-like)")
    ax.set_title("(c) What missed phishing lacks", loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    style(ax, grid="x")

    fig.tight_layout(w_pad=1.5)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_error_analysis.png", dpi=300)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va, te, feats = load()
    df = build_table(tr, va, te, feats)
    pfe = per_feature(feats)
    tau = df["tau"].iloc[0]

    errs = df[df["outcome"].isin(["FN", "FP"])].copy()
    errs.to_csv(OUT / "error_all_cases.csv")

    bd = breakdown(df)
    prof = feature_profile(df, te, tr, feats)
    picks = select_cases(df)
    cases, detail = case_table(df, picks, pfe, te, feats)
    counts = df["outcome"].value_counts()

    md = [
        "# Group C — FP / FN analysis of the Tier 2 autoencoder", "",
        f"Run `t2_ae__feat-all__cont-0__score-ce__tau-f1`, seed {SEED}, test split (735 phishing, 924 legit). "
        f"τ = {tau:.3f} nats (best F1 on validation). "
        f"TP {counts['TP']} · FP {counts['FP']} · FN {counts['FN']} · TN {counts['TN']}.", "",
        "## Error breakdown", "", bd.to_markdown(), "",
        "Near miss = s(x) within a factor 2 of τ; inversion = off by a factor ≥ 10. "
        "'XGBoost gets it right' = the supervised Tier 3 model (seed 0) classifies the site correctly.", "",
        "## Feature profile (share of sites with the feature coded −1)", "",
        prof.sort_values("TP − FN", ascending=False).head(10).round(2).to_markdown(), "",
        "## Case studies", "", f"τ = {tau:.3f}. s_j = per-feature term −log p̂_j(x_j); the 30 terms add up to s(x).", "",
        cases, "", detail, "",
    ]
    (OUT / "error_analysis.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    figure(df, prof)


if __name__ == "__main__":
    main()
