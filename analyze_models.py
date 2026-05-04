"""
Comparative analysis: DeepSeek V3.2 vs Grok 4.1 Fast — Dictator Game.
Produces all thesis figures (two-model comparison version).

Usage: python analyze_models.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "out"

# Bechler (2015) human DG data (digitized from Fig. 1)
HUMAN_DG = pd.DataFrame({
    "social_distance": [2, 20, 100, 2, 20, 100, 2, 20, 100],
    "money": [10, 10, 10, 3000, 3000, 3000, 250000, 250000, 250000],
    "share_pct": [0.41, 0.19, 0.12, 0.29, 0.10, 0.05, 0.25, 0.05, 0.025],
})

MODEL_DEEPSEEK = "deepseek/deepseek-v3.2"
MODEL_GROK = "x-ai/grok-4.1-fast"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_model_dg(model_id: str) -> pd.DataFrame:
    """Load main data for a given model. Prefers v2 files, falls back to v1."""
    model_short = "deepseek" if "deepseek" in model_id else "grok"

    # v2 (new experiments): single combined file
    v2_csv = OUT_DIR / f"results_v2_{model_short}.csv"

    # v1 fallback: main + extended split files
    if "deepseek" in model_id:
        v1_csvs = [OUT_DIR / "results_main_n3.csv", OUT_DIR / "results_main_extended.csv"]
    else:
        v1_csvs = [OUT_DIR / "results_grok_main.csv", OUT_DIR / "results_grok_extended.csv"]

    frames = []
    if v2_csv.exists():
        df = pd.read_csv(v2_csv)
        if "model" not in df.columns:
            df["model"] = model_id
        if "game" not in df.columns:
            df["game"] = "dictator"
        frames.append(df)
        print(f"  Loaded v2: {v2_csv.name} ({len(df)} rows)")
    else:
        for p in v1_csvs:
            if p.exists():
                df = pd.read_csv(p)
                if "model" not in df.columns:
                    df["model"] = model_id
                if "game" not in df.columns:
                    df["game"] = "dictator"
                frames.append(df)
        if frames:
            print(f"  Note: using v1 data for {model_short} (v2 not found)")

    if not frames:
        return pd.DataFrame(columns=["social_distance", "money", "share_pct"])

    df = pd.concat(frames, ignore_index=True)
    df = df[
        (df["model"] == model_id)
        & (df["game"] == "dictator")
        & df["share_pct"].notna()
    ].copy()
    for col in ["social_distance", "money", "share_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["social_distance", "money", "share_pct"])


def load_robustness_model(model: str, condition: str) -> pd.DataFrame:
    """Load robustness CSV for given model and condition name. Prefers v2 files."""
    model_short = "deepseek" if "deepseek" in model else "grok"

    # v2 file takes priority
    v2_path = OUT_DIR / f"results_v2_robust_{model_short}_{condition}.csv"

    # v1 fallback
    if "deepseek" in model:
        v1_paths = [OUT_DIR / f"results_robust_{condition}.csv",
                    OUT_DIR / f"results_robust_{condition}_ext.csv"]
    else:
        v1_paths = [OUT_DIR / f"results_grok_robust_{condition}.csv"]

    frames = []
    if v2_path.exists():
        df = pd.read_csv(v2_path)
        if "model" not in df.columns:
            df["model"] = model
        if "game" not in df.columns:
            df["game"] = "dictator"
        frames.append(df)
    else:
        for p in v1_paths:
            if p.exists():
                df = pd.read_csv(p)
                if "model" not in df.columns:
                    df["model"] = model
                if "game" not in df.columns:
                    df["game"] = "dictator"
                frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["social_distance", "money", "share_pct"])

    df = pd.concat(frames, ignore_index=True)
    for col in ["social_distance", "money", "share_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["social_distance", "money", "share_pct"])


# --------------------------------------------------------------------------- #
# OLS regression
# --------------------------------------------------------------------------- #

def fit_ols(y: np.ndarray, X: np.ndarray) -> dict:
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = n - p
    s2 = float((resid**2).sum()) / df
    xtxi = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(s2 * xtxi))
    t = beta / se
    tcrit = 1.96
    r2 = 1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()
    return {
        "n": n, "r2": float(r2), "beta": beta, "se": se, "t": t,
        "ci_lo": beta - tcrit * se, "ci_hi": beta + tcrit * se,
        "sigma": math.sqrt(s2),
    }


def regression_results(df: pd.DataFrame, label: str = "") -> dict:
    """Return dict of OLS results (raw + log stake) for a model's cell means."""
    cells = df.groupby(["social_distance", "money"])["share_pct"].mean().reset_index()
    y = cells["share_pct"].to_numpy()
    sd = cells["social_distance"].to_numpy()
    stake = cells["money"].to_numpy()
    results = {}
    for name, s_var in [("raw", stake), ("log", np.log(stake))]:
        X = np.column_stack([np.ones(len(y)), sd, s_var])
        results[name] = fit_ols(y, X)
    if label:
        print(f"\n=== OLS regression — {label} ===")
        for name, r in results.items():
            stake_label = "raw stake ($)" if name == "raw" else "log(stake)"
            print(f"  [{stake_label}] n={r['n']}, R²={r['r2']:.3f}")
            coef_names = ["intercept", "social_distance", "stake"]
            for i, cn in enumerate(coef_names):
                ci = f"[{r['ci_lo'][i]:.4g}, {r['ci_hi'][i]:.4g}]"
                print(f"    {cn:<18} {r['beta'][i]:>10.4g}  t={r['t'][i]:>7.2f}  {ci}")
    return results


def robustness_regression(df: pd.DataFrame) -> dict:
    """OLS on cell means for a robustness dataframe."""
    if df.empty:
        return {}
    cells = df.groupby(["social_distance", "money"])["share_pct"].mean().reset_index()
    y = cells["share_pct"].to_numpy()
    sd = cells["social_distance"].to_numpy()
    stake = cells["money"].to_numpy()
    X = np.column_stack([np.ones(len(y)), sd, np.log(stake)])
    return fit_ols(y, X)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _save(name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def _cell_mean(df: pd.DataFrame, money: int | None = None,
               sd: int | None = None) -> pd.Series:
    sub = df.copy()
    if money is not None:
        sub = sub[sub["money"] == money]
    if sd is not None:
        sub = sub[sub["social_distance"] == sd]
    grp = "social_distance" if money is not None else "money"
    return sub.groupby(grp)["share_pct"].mean()


# --------------------------------------------------------------------------- #
# Figure 1: SD effect — both models + humans (side by side)
# --------------------------------------------------------------------------- #

def fig_sd_effect(ds: pd.DataFrame, gk: pd.DataFrame) -> None:
    """
    2-panel figure: SD effect at stake=$10.
    Left = DeepSeek, Right = Grok. Both overlaid with human data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    stake = 10

    for ax, df, title in zip(
        axes,
        [ds, gk],
        ["(a) DeepSeek V3.2", "(b) Grok 4.1 Fast"],
    ):
        llm_sd = _cell_mean(df, money=stake)
        ax.plot(llm_sd.index, llm_sd.values,
                marker="o", markersize=5, linewidth=1.8, color="steelblue",
                label="LLM (stake = $10)")
        h10 = HUMAN_DG[HUMAN_DG["money"] == 10].sort_values("social_distance")
        ax.plot(h10["social_distance"], h10["share_pct"],
                marker="s", markersize=8, linewidth=1.5, linestyle="--",
                color="coral", label="Humans (Bechler 2015)")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Social distance (rank)")
        ax.set_ylim(0, 0.70)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Transferred share (fraction of stake)")
    _save("fig_sd_effect.png")


# --------------------------------------------------------------------------- #
# Figure 2: Stake effect — both models + humans (side by side)
# --------------------------------------------------------------------------- #

def fig_stake_effect(ds: pd.DataFrame, gk: pd.DataFrame) -> None:
    """
    2-panel figure: stake effect at SD=20, full $1–$250K range.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    sd_val = 20

    for ax, df, title in zip(
        axes,
        [ds, gk],
        ["(a) DeepSeek V3.2", "(b) Grok 4.1 Fast"],
    ):
        llm_s = _cell_mean(df, sd=sd_val)
        ax.plot(llm_s.index, llm_s.values,
                marker="o", markersize=4, linewidth=1.8, color="steelblue",
                label="LLM (SD = 20)")
        h20 = HUMAN_DG[HUMAN_DG["social_distance"] == 20].sort_values("money")
        ax.scatter(h20["money"], h20["share_pct"],
                   marker="s", s=80, color="coral", zorder=5,
                   label="Humans (Bechler 2015)")
        ax.set_xscale("log")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Stake (dollars, log scale)")
        ax.set_ylim(0, 0.70)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Transferred share (fraction of stake)")
    _save("fig_stake_effect.png")


# --------------------------------------------------------------------------- #
# Figure 3: Heatmaps side by side
# --------------------------------------------------------------------------- #

def fig_heatmaps(ds: pd.DataFrame, gk: pd.DataFrame) -> None:
    """2-panel heatmap: DeepSeek (left) and Grok (right)."""
    if gk.empty:
        # single-model fallback
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        _heatmap_panel(ax, ds, "DeepSeek V3.2")
        _save("fig_heatmaps.png")
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    for ax, df, title in zip(axes, [ds, gk],
                              ["(a) DeepSeek V3.2", "(b) Grok 4.1 Fast"]):
        _heatmap_panel(ax, df, title)
    plt.subplots_adjust(wspace=0.05)
    _save("fig_heatmaps.png")


def _heatmap_panel(ax, df: pd.DataFrame, title: str) -> None:
    pivot = df.pivot_table(
        index="social_distance", columns="money",
        values="share_pct", aggfunc="mean",
    )
    im = ax.imshow(
        pivot.values, aspect="auto", cmap="RdYlGn",
        vmin=0, vmax=0.55, origin="lower",
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [f"${c:,}" if c >= 1000 else f"${c}" for c in pivot.columns],
        rotation=90, fontsize=5,
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index], fontsize=7)
    ax.set_xlabel("Stake ($)", fontsize=9)
    ax.set_ylabel("Social distance", fontsize=9)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label="Mean share", fraction=0.025, pad=0.02)


# --------------------------------------------------------------------------- #
# Figure 4: Robustness comparison (one panel per model)
# --------------------------------------------------------------------------- #

def fig_robustness(ds: pd.DataFrame, gk: pd.DataFrame) -> None:
    """2-panel robustness: each panel = one model, 3 conditions + main + humans."""
    # SD=5 removed: not in v2 grid ([1,2,4,6,...,100])
    sds = [1, 2, 10, 20, 50, 100]
    money_val = 10
    conditions = [
        ("temp_low",   "Low temp (0.3)",   "darkorange"),
        ("temp_high",  "High temp (1.5)",  "green"),
        ("altprompt",  "Alt. prompt",      "purple"),
    ]

    n_panels = 2 if not gk.empty else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(12 if n_panels == 2 else 7, 5),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    datasets = [(ds, MODEL_DEEPSEEK, "(a) DeepSeek V3.2"),
                (gk, MODEL_GROK,     "(b) Grok 4.1 Fast")][:n_panels]

    for ax, (df_main, model_id, title) in zip(axes, datasets):
        sub = df_main[df_main["money"] == money_val].groupby("social_distance")["share_pct"].mean()
        ax.plot(sds, sub.reindex(sds).values,
                marker="o", linewidth=2, label="Main (default temp)", color="steelblue")

        for cond, label, color in conditions:
            rdf = load_robustness_model(model_id, cond)
            if rdf.empty:
                continue
            rsub = rdf[rdf["money"] == money_val].groupby("social_distance")["share_pct"].mean()
            ax.plot(sds, rsub.reindex(sds).values,
                    marker="s", linewidth=1.8, label=label, color=color)

        h10 = HUMAN_DG[HUMAN_DG["money"] == 10].sort_values("social_distance")
        ax.scatter(h10["social_distance"], h10["share_pct"],
                   marker="^", s=100, color="red", zorder=6,
                   label="Humans (Bechler 2015)")

        ax.set_xlabel("Social distance")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.75)

    axes[0].set_ylabel("Transferred share (stake = $10)")
    _save("fig_robustness.png")


# --------------------------------------------------------------------------- #
# Figure 5 (Appendix): Regression scatter — both models
# --------------------------------------------------------------------------- #

def fig_regression_scatter(ds: pd.DataFrame, gk: pd.DataFrame) -> None:
    """Scatter of mean share vs SD with OLS line, both models on one plot."""
    fig, ax = plt.subplots(figsize=(7, 4))

    colors = {"DeepSeek V3.2": "steelblue", "Grok 4.1 Fast": "darkorange"}
    for df, name, color in [(ds, "DeepSeek V3.2", "steelblue"),
                             (gk, "Grok 4.1 Fast", "darkorange")]:
        if df.empty:
            continue
        m = df.groupby("social_distance")["share_pct"].mean().reset_index()
        x, y = m["social_distance"].to_numpy(), m["share_pct"].to_numpy()
        ax.scatter(x, y, color=color, s=25, zorder=5, label=f"{name} (cell means)")
        X = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        xl = np.linspace(x.min(), x.max(), 200)
        ax.plot(xl, beta[0] + beta[1] * xl, color=color, linewidth=2,
                linestyle="--",
                label=f"{name}: {beta[0]:.3f} {beta[1]:+.5f}×SD")

    ax.set_xlabel("Social distance (rank)")
    ax.set_ylabel("Mean transferred share")
    ax.set_title("OLS regression: share on social distance (both models)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.65)
    _save("fig_regression_scatter.png")


# --------------------------------------------------------------------------- #
# Print key stats
# --------------------------------------------------------------------------- #

def print_stats(df: pd.DataFrame, label: str) -> None:
    cells = df.groupby(["social_distance", "money"])["share_pct"].mean().reset_index()
    n_cells = len(cells)
    print(f"\n=== {label} ===")
    print(f"  Obs: {len(df)}, cells: {n_cells}")
    print(f"  SD range: {sorted(df.social_distance.unique())[:3]}...{sorted(df.social_distance.unique())[-3:]}")
    print(f"  Stake range: ${df.money.min():.0f}–${df.money.max():.0f}")
    by_sd = df.groupby("social_distance")["share_pct"].mean()
    print(f"  Share at SD=1: {by_sd.get(1, float('nan')):.3f}")
    print(f"  Share at SD=100: {by_sd.get(100, float('nan')):.3f}")
    print(f"  Mean over all: {df.share_pct.mean():.3f}")
    by_stake = df.groupby("money")["share_pct"].mean()
    print(f"  Share at stake=$10: {by_stake.get(10, float('nan')):.3f}")
    if 250000 in by_stake.index:
        print(f"  Share at stake=$250K: {by_stake.get(250000, float('nan')):.3f}")


# --------------------------------------------------------------------------- #
# Robustness table helper
# --------------------------------------------------------------------------- #

def print_robustness_table(ds: pd.DataFrame, gk: pd.DataFrame) -> None:
    """Print robustness regression table for both models."""
    conditions = [
        ("main",      "Main (default)", None,  None, False),
        ("temp_low",  "Low temp (0.3)", 0.3,   None, False),
        ("temp_high", "High temp (1.5)", 1.5,  None, False),
        ("altprompt", "Alt. prompt",    None,  None, True),
    ]
    print("\n=== Robustness regression table (OLS, log stake, cell means) ===")
    print(f"  {'Condition':<22} {'Model':<18} {'c0':>7} {'c_SD':>9} {'c_log':>9} "
          f"{'t_SD':>6} {'R2':>6} {'N':>5}")
    print("  " + "-" * 82)

    for cond_name, cond_label, *_ in conditions:
        for model_id, df_main in [(MODEL_DEEPSEEK, ds), (MODEL_GROK, gk)]:
            model_short = "DeepSeek" if "deepseek" in model_id else "Grok"
            if cond_name == "main":
                df = df_main
            else:
                df = load_robustness_model(model_id, cond_name)
            if df.empty:
                print(f"  {cond_label:<22} {model_short:<18} (no data)")
                continue
            r = robustness_regression(df)
            if not r:
                continue
            c0, c_sd, c_log = r["beta"]
            t_sd = r["t"][1]
            print(f"  {cond_label:<22} {model_short:<18} "
                  f"{c0:>7.4f} {c_sd:>9.5f} {c_log:>9.5f} "
                  f"{t_sd:>6.2f} {r['r2']:>6.3f} {r['n']:>5}")


# --------------------------------------------------------------------------- #
# Generosity comparison (one-sample t-tests vs Bechler 2015 benchmarks)
# --------------------------------------------------------------------------- #

def print_generosity_stats(ds: pd.DataFrame, gk: pd.DataFrame) -> None:
    """
    One-sample t-tests: LLM cell means vs Bechler et al. human benchmarks.
    Averaged over stakes in $10–$200 range at SD=20 and SD=100.
    Bechler benchmarks: 11.5% at SD=20, 6.5% at SD=100 (avg over $10–$200).
    """
    human_benchmarks = {20: 0.115, 100: 0.065}

    print("\n=== Generosity comparison: LLMs vs. Bechler et al. (2015) ===")
    print(f"  Method: one-sample t-test of LLM cell means against human benchmark.")
    print(f"  Stakes averaged: $10–$200 range at given SD.")
    print(f"  H0: LLM mean == human benchmark; t = (mean_LLM - benchmark) / SE\n")

    for sd_val, human_bench in human_benchmarks.items():
        print(f"  SD={sd_val} (human benchmark = {human_bench*100:.1f}%)")
        for label, df in [("DeepSeek", ds), ("Grok", gk)]:
            if df.empty:
                continue
            sub = df[(df["social_distance"] == sd_val) & (df["money"] <= 200) & (df["money"] >= 10)]
            if sub.empty:
                print(f"    {label}: no data")
                continue
            # Cell means in this SD/stake slice
            cell_means = sub.groupby("money")["share_pct"].mean().to_numpy(dtype=float)
            n = len(cell_means)
            mean_val = cell_means.mean()
            se = cell_means.std(ddof=1) / (n ** 0.5) if n > 1 else float("nan")
            t_stat = (mean_val - human_bench) / se if se > 0 else float("nan")
            # Approximate p-value (two-sided, large-n normal approximation)
            import math
            p_approx = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2)))) if not math.isnan(t_stat) else float("nan")
            stars = "***" if p_approx < 0.001 else ("**" if p_approx < 0.01 else ("*" if p_approx < 0.05 else ""))
            print(f"    {label}: mean={mean_val*100:.1f}%, n={n} cells, "
                  f"t={t_stat:.1f}, p={p_approx:.4f} {stars}")
        print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ds = load_model_dg(MODEL_DEEPSEEK)
    gk = load_model_dg(MODEL_GROK)

    print_stats(ds, "DeepSeek V3.2")
    if not gk.empty:
        print_stats(gk, "Grok 4.1 Fast")

    regression_results(ds, "DeepSeek V3.2")
    if not gk.empty:
        regression_results(gk, "Grok 4.1 Fast")

    print_robustness_table(ds, gk)
    print_generosity_stats(ds, gk)

    print("\nGenerating figures...")
    fig_sd_effect(ds, gk)
    fig_stake_effect(ds, gk)
    fig_heatmaps(ds, gk)
    fig_robustness(ds, gk)
    fig_regression_scatter(ds, gk)
    print("\nDone.")


if __name__ == "__main__":
    main()
