"""Group C: interpretability and practical impact of the Tier 2 autoencoder (paper section 8.2).

  python "part c/src/interpretability.py"

Model: Group B's final run t2_ae__feat-all__cont-0__score-ce__tau-f1, seed 0, test split.
Explanation = the score's own decomposition s(x) = sum_j s_j(x), s_j = -log p_j(x_j).

1. Local:  why one typical alert was raised (per-feature terms of the median-score TP).
2. Global: which features drive alerts (mean share of s(x) over all alerts, every seed),
           compared with supervised XGBoost's SHAP values (pred_contribs) and Group A's MI ranking.
3. Faithfulness (deletion test): replace the k features the explanation blames with the most
   common legitimate value and re-score with the saved model. If the explanation is faithful,
   alerts clear much faster than when k random features are replaced.
4. Practical impact: alerts / catches / false alarms per 10,000 sites at the dataset's
   phishing share and at a realistic 1% share, for each tier.

Writes part c/results/interpretability.md, interp_*.csv and part c/figures/fig_interpretability.png.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HW2 = Path(__file__).resolve().parents[2]
PB = HW2 / "part b"
sys.path.insert(0, str(PB / "src"))
import torch  # noqa: E402
import xgboost as xgb  # noqa: E402
from common import MI_RANK, load_json  # noqa: E402
from methods import CategoricalAutoencoder, _AENet  # noqa: E402

OUT, FIG = HW2 / "part c" / "results", HW2 / "part c" / "figures"
RUNS = PB / "results" / "runs"
AE_RUN = RUNS / "t2_ae__feat-all__cont-0__score-ce__tau-f1"
XGB_RUN = RUNS / "t3_xgb__feat-all__cont-0__score-ce__tau-f1"
SEED = 0
REAL_PREVALENCE = 0.01


def load_ae(feats, seed):
    params = load_json(AE_RUN / "metrics.json")["params"]
    m = CategoricalAutoencoder(hidden=params["hidden"], latent=params["latent"])
    m.net = _AENet(len(feats), tuple(params["hidden"]), params["latent"])
    m.net.load_state_dict(torch.load(AE_RUN / "models" / f"seed{seed}.pt"))
    m.net.eval()
    return m


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tr, te = (pd.read_csv(PB / "data" / f"{s}.csv") for s in ("train", "test"))
    feats = [c for c in tr.columns if c not in ("row_id", "Result")]
    te = te.set_index("row_id")
    pred = pd.read_csv(AE_RUN / "predictions.csv")
    pred = pred[pred["split"] == "test"]
    p0 = pred[pred["seed"] == SEED].set_index("row_id")
    tau = float(p0["tau"].iloc[0])
    pfe_all = pd.read_csv(AE_RUN / "per_feature_error.csv")
    pfe = pfe_all[pfe_all["seed"] == SEED].set_index("row_id")[feats]
    alerts = p0.index[p0["pred"] == 1]
    tp = p0.index[(p0["pred"] == 1) & (p0["is_phishing"] == 1)]
    md = ["# Group C — interpretability and practical impact (Tier 2 autoencoder)", "",
          f"Run t2_ae (Group B final), seed {SEED}, test split. τ = {tau:.3f} nats.", ""]

    # ---------------------------------------------------------------- 1. local explanation
    tp_scores = p0.loc[tp, "score"].sort_values()
    rid = tp_scores.index[len(tp_scores) // 2]                     # median-score true alert
    terms = pfe.loc[rid].sort_values(ascending=False)
    share = terms / terms.sum()
    legit = tr[tr["Result"] == 1]
    local = pd.DataFrame({"code": te.loc[rid, terms.index].astype(int), "s_j": terms, "share": share,
                          "P(code | legit train)": [(legit[f] == te.loc[rid, f]).mean() for f in terms.index]})
    local.to_csv(OUT / "interp_local_example.csv")
    md += [f"## 1. Why was {rid} flagged? (median-score true alert)", "",
           f"s(x) = {p0.loc[rid, 'score']:.2f} nats = {p0.loc[rid, 'score'] / tau:.1f} × τ. "
           f"Top 3 features carry {share.iloc[:3].sum():.0%} of the score; top 5 carry {share.iloc[:5].sum():.0%}.", "",
           local.head(8).round(3).to_markdown(), ""]

    # ---------------------------------------------------------------- 2. global attribution
    rows = []
    for seed, g in pfe_all.groupby("seed"):
        ps = pred[pred["seed"] == seed].set_index("row_id")
        a = ps.index[ps["pred"] == 1]
        t = g.set_index("row_id").loc[a, feats]
        rows.append((t.div(t.sum(axis=1), axis=0)).mean())          # mean share of s(x) per feature
    ae_share = pd.concat(rows, axis=1)
    glob = pd.DataFrame({"AE share of alert score": ae_share.mean(axis=1), "AE std over seeds": ae_share.std(axis=1)})

    booster = xgb.Booster()
    booster.load_model(str(XGB_RUN / "models" / f"seed{SEED}.json"))
    contrib = booster.predict(xgb.DMatrix(te[feats].to_numpy(dtype=float), feature_names=feats),
                              pred_contribs=True)[:, :-1]
    shap_abs = np.abs(contrib).mean(axis=0)
    glob["XGB mean |SHAP| share"] = shap_abs / shap_abs.sum()
    glob["MI rank (Group A)"] = [MI_RANK.index(f) + 1 for f in glob.index]
    glob = glob.sort_values("AE share of alert score", ascending=False)
    glob["AE rank"] = np.arange(1, len(glob) + 1)
    glob["XGB rank"] = glob["XGB mean |SHAP| share"].rank(ascending=False).astype(int)
    glob.to_csv(OUT / "interp_global.csv")
    rho_x = spearmanr(glob["AE rank"], glob["XGB rank"])
    rho_m = spearmanr(glob["AE rank"], glob["MI rank (Group A)"])
    rho_xm = spearmanr(glob["XGB rank"], glob["MI rank (Group A)"])
    top5 = lambda col: set(glob.sort_values(col).index[:5])  # noqa: E731
    overlap = top5("AE rank") & top5("XGB rank")
    # local agreement for the example
    ex_shap = pd.Series(contrib[te.index.get_loc(rid)], index=feats).sort_values(ascending=False)
    md += ["## 2. What drives alerts globally?", "",
           "AE share = mean fraction of s(x) contributed by each feature over all alerts (TP+FP), "
           "averaged over seeds 0–4 (± std over seeds). XGB = mean |SHAP| (log-odds) over all test sites.", "",
           glob.head(10).round(3).to_markdown(), "",
           f"Spearman rank correlation over 30 features: AE vs XGB ρ = {rho_x.statistic:.2f} (p = {rho_x.pvalue:.2g}); "
           f"AE vs MI ρ = {rho_m.statistic:.2f} (p = {rho_m.pvalue:.2g}); XGB vs MI ρ = {rho_xm.statistic:.2f} "
           f"(p = {rho_xm.pvalue:.2g}). Top-5 overlap AE∩XGB: {sorted(overlap)}.", "",
           f"For {rid}, XGB's top SHAP features (toward phishing): "
           + ", ".join(f"{f} {v:+.2f}" for f, v in ex_shap.head(4).items())
           + f"; XGB P(phishing) = {1 / (1 + np.exp(-contrib[te.index.get_loc(rid)].sum() - booster.predict(xgb.DMatrix(te.loc[[rid], feats].to_numpy(dtype=float), feature_names=feats), pred_contribs=True)[0, -1])):.3f}.", ""]

    # ---------------------------------------------------------------- 3. deletion test
    model = load_ae(feats, SEED)
    mode = legit[feats].mode().iloc[0].astype(int)
    X = te.loc[tp, feats].to_numpy().astype(np.int8)          # true alerts only
    order = np.argsort(-pfe.loc[tp, feats].to_numpy(), axis=1)
    rng = np.random.default_rng(0)
    ks = [0, 1, 2, 3, 5]
    dele = []
    for k in ks:
        Xa = X.copy()
        for i in range(len(Xa)):
            Xa[i, order[i, :k]] = mode.to_numpy()[order[i, :k]]
        cleared_attr = (model.score(Xa) < tau).mean()
        rnd = []
        for rep in range(20):
            Xr = X.copy()
            for i in range(len(Xr)):
                idx = rng.choice(len(feats), size=k, replace=False)
                Xr[i, idx] = mode.to_numpy()[idx]
            rnd.append((model.score(Xr) < tau).mean())
        dele.append({"k features neutralised": k, "alerts cleared (explanation top-k)": cleared_attr,
                     "alerts cleared (random k), mean": np.mean(rnd), "random k, std": np.std(rnd)})
    dele = pd.DataFrame(dele)
    dele.to_csv(OUT / "interp_deletion.csv", index=False)
    md += ["## 3. Faithfulness: deletion test", "",
           f"For each of the {len(tp)} true alerts, the k features with the largest s_j are set to their most "
           "common legitimate value and the site is re-scored by the saved seed-0 model. "
           "Random baseline: k random features, 20 repeats.", "", dele.round(3).to_markdown(index=False), ""]

    # ---------------------------------------------------------------- 4. practical impact
    imp = []
    names = {"t1_count": "T1 count of −1", "t2_iforest": "T2 alt. Isolation Forest",
             "t2_ae": "T2 autoencoder (ours)", "t3_xgb": "T3 XGBoost (supervised)"}
    for m, nm in names.items():
        mt = load_json(RUNS / f"{m}__feat-all__cont-0__score-ce__tau-f1" / "metrics.json")["summary"]
        rec, fpr = mt["test_recall"]["mean"], mt["test_fpr"]["mean"]
        for pi, n in ((0.443, 10_000), (REAL_PREVALENCE, 10_000)):
            ph = n * pi
            caught, fa = rec * ph, fpr * (n - ph)
            imp.append({"method": nm, "phishing share": f"{pi:.1%}", "phishing sites": round(ph),
                        "caught": round(caught), "missed": round(ph - caught), "false alarms": round(fa),
                        "alerts to review": round(caught + fa), "precision": caught / (caught + fa),
                        "false alarms per catch": fa / caught})
    imp = pd.DataFrame(imp)
    imp.to_csv(OUT / "interp_impact.csv", index=False)
    md += ["## 4. Practical impact per 10,000 scanned sites", "",
           "Recall and FPR at τ from Group B's main table (mean over 5 seeds), scaled to 10,000 sites.", "",
           imp.round(2).to_markdown(index=False), ""]
    (OUT / "interpretability.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    figure(local, glob, dele, rid, p0.loc[rid, "score"], tau)


BLUE, ORANGE, AQUA, INK, MUTED, GRID, GRAY = ("#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e",
                                              "#e4e3df", "#9a9993")


def style(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure(local, glob, dele, rid, score, tau):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 9, "axes.labelsize": 8,
                         "axes.labelcolor": INK})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.1, 1.2, 0.85]})

    # (a) local: per-feature terms of the example alert
    ax = axes[0]
    top = local.head(6)
    rest = local["s_j"].iloc[6:].sum()
    labels = [f"{f} = {c:+d}" for f, c in zip(top.index, top["code"])] + ["other 24 features"]
    vals = list(top["s_j"]) + [rest]
    yi = np.arange(len(vals))[::-1]
    ax.barh(yi, vals, color=[BLUE] * len(top) + [GRAY], height=0.6)
    for y, v in zip(yi, vals):
        ax.text(v + score * 0.01, y, f"{v:.1f} ({v / score:.0%})", va="center", fontsize=7, color=INK)
    ax.set_yticks(yi, labels, fontsize=7.5)
    ax.set_xlim(0, max(vals) * 1.35)
    ax.set_xlabel("per-feature term s_j(x) = −log p̂_j (nats)")
    ax.set_title(f"(a) Why {rid} was flagged\n     s(x) = {score:.1f} nats, τ = {tau:.1f}", loc="left")
    style(ax)

    # (b) global: AE share vs XGB SHAP share, top 8 features by AE
    ax = axes[1]
    g = glob.head(8).iloc[::-1]
    yi = np.arange(len(g))
    ax.scatter(g["AE share of alert score"], yi + 0.12, color=BLUE, s=30, marker="o",
               label="AE score share", zorder=3)
    ax.errorbar(g["AE share of alert score"], yi + 0.12, xerr=g["AE std over seeds"], fmt="none",
                ecolor=BLUE, elinewidth=1, capsize=2)
    ax.scatter(g["XGB mean |SHAP| share"], yi - 0.12, color=ORANGE, s=30, marker="s",
               label="XGB |SHAP| share", zorder=3)
    ax.set_yticks(yi, [f"{f} (MI #{r})" for f, r in zip(g.index, g["MI rank (Group A)"])], fontsize=7.5)
    ax.set_xlabel("share of attribution")
    ax.set_title("(b) What drives alerts (all alerts, 5 seeds)", loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    style(ax)

    # (c) deletion test
    ax = axes[2]
    k = dele["k features neutralised"]
    ax.plot(k, dele["alerts cleared (explanation top-k)"], color=BLUE, marker="o", lw=2, ms=5,
            label="explanation top-k")
    ax.errorbar(k, dele["alerts cleared (random k), mean"], yerr=dele["random k, std"], color=GRAY,
                marker="D", lw=2, ms=4, capsize=2, label="random k")
    ax.set_xticks(k)
    ax.set_ylim(0, 1)
    ax.set_xlabel("k features reset\nto legit value")
    ax.set_ylabel("share of true alerts cleared (s < τ)")
    ax.set_title("(c) Deletion test", loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    style(ax, grid="y")

    fig.tight_layout(w_pad=1.5)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_interpretability.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


if __name__ == "__main__":
    main()
