from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path("/home/demid/Научная работа с Дагаевым/exp")
OUT_DIR = BASE_DIR / "out"


LLM_MODELS = [
    "google/gemini-3-flash-preview",
    "deepseek/deepseek-v3.2",
    "x-ai/grok-4.1-fast",
    "anthropic/claude-sonnet-4.5",
]


SOCIAL_DISTANCES = [2, 20, 100]
MONEY_AMOUNTS = [10, 3000, 250_000]


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")


def load_llm_means(game: str) -> pd.DataFrame:
    paths = sorted(OUT_DIR.glob(f"results_*_{game}.csv"))
    if not paths:
        raise FileNotFoundError(f"No results_*_{game}.csv in {OUT_DIR}")

    dfs: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        if "share" not in df.columns:
            raise ValueError(f"Missing share column in {p.name}")
        if "money" not in df.columns:
            raise ValueError(f"Missing money column in {p.name}")
        df["share_pct"] = df["share"] / df["money"].astype(float)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df[all_df["model"].isin(LLM_MODELS)].copy()
    all_df = all_df[all_df["social_distance"].isin(SOCIAL_DISTANCES)].copy()
    all_df = all_df[all_df["money"].isin(MONEY_AMOUNTS)].copy()

    grp = ["model", "game", "money", "social_distance"]
    out = (
        all_df.groupby(grp, dropna=False)["share_pct"]
        .mean()
        .reset_index(name="share_pct_mean")
    )
    return out


def human_digitized() -> pd.DataFrame:
    """
    Approximate values digitized by eye from the user-provided human plot.

    Columns match LLM aggregate: game, money, social_distance, share_pct_mean.
    """
    # Dictator (left panel): Proportion Offered
    d = {
        10: {2: 0.41, 20: 0.19, 100: 0.12},
        3000: {2: 0.29, 20: 0.10, 100: 0.05},
        250_000: {2: 0.25, 20: 0.05, 100: 0.025},
    }

    # Ultimatum (right panel): Proportion Offered
    u = {
        10: {2: 0.47, 20: 0.35, 100: 0.26},
        3000: {2: 0.38, 20: 0.23, 100: 0.16},
        250_000: {2: 0.34, 20: 0.19, 100: 0.14},
    }

    rows = []
    for money, xs in d.items():
        for x, v in xs.items():
            rows.append(
                {
                    "model": "humans",
                    "game": "dictator",
                    "money": int(money),
                    "social_distance": int(x),
                    "share_pct_mean": float(v),
                }
            )

    for money, xs in u.items():
        for x, v in xs.items():
            rows.append(
                {
                    "model": "humans",
                    "game": "ultimatum",
                    "money": int(money),
                    "social_distance": int(x),
                    "share_pct_mean": float(v),
                }
            )

    return pd.DataFrame(rows)


def _save(fig: plt.Figure, fname: str) -> Path:
    path = OUT_DIR / fname
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sonnet_vs_humans(df_llm: pd.DataFrame, df_h: pd.DataFrame,
                          game: str) -> Path:
    title = "Sonnet-4.5 vs Real Humans"
    df_s = df_llm[(df_llm["game"] == game)
                  & (df_llm["model"] == "anthropic/claude-sonnet-4.5")]
    df_s = df_s.copy()
    df_hg = df_h[df_h["game"] == game].copy()

    colors = {10: "#1f77b4", 3000: "#ff7f0e", 250_000: "#2ca02c"}

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for money in MONEY_AMOUNTS:
        c = colors[int(money)]
        ds = df_s[df_s["money"] == money].sort_values("social_distance")
        dh = df_hg[df_hg["money"] == money].sort_values("social_distance")

        ax.plot(
            ds["social_distance"],
            ds["share_pct_mean"],
            color=c,
            marker="o",
            linestyle="-",
            label=f"Sonnet m=${money}",
        )
        ax.plot(
            dh["social_distance"],
            dh["share_pct_mean"],
            color=c,
            marker="o",
            linestyle="--",
            label=f"Humans m=${money}",
            alpha=0.9,
        )

    ax.set_title(f"{title} ({game})")
    ax.set_xlabel("social distance")
    ax.set_ylabel("transfered share")
    ax.set_ylim(-0.02, 0.55)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

    return _save(fig, f"{_slug(game)}_{_slug(title)}.png")


def plot_llms_vs_humans(df_llm: pd.DataFrame, df_h: pd.DataFrame, game: str,
                        money: int) -> Path:
    title = "LLMs vs Real Humans comparison"
    df_g = df_llm[(df_llm["game"] == game) & (df_llm["money"] == money)].copy()
    df_hm = df_h[(df_h["game"] == game) & (df_h["money"] == money)].copy()

    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    order = ["humans"] + LLM_MODELS
    colors = {
        "humans": "#000000",
        "anthropic/claude-sonnet-4.5": "#d62728",
        "deepseek/deepseek-v3.2": "#1f77b4",
        "google/gemini-3-flash-preview": "#2ca02c",
        "x-ai/grok-4.1-fast": "#ff7f0e",
    }

    for model in order:
        if model == "humans":
            d = df_hm.sort_values("social_distance")
        else:
            d = df_g[df_g["model"] == model].sort_values("social_distance")

        if d.empty:
            continue

        ax.plot(
            d["social_distance"],
            d["share_pct_mean"],
            marker="o",
            linewidth=2.0 if model == "humans" else 1.8,
            linestyle="--" if model == "humans" else "-",
            color=colors.get(model, None),
            label=model,
        )

    ax.set_title(f"{title} ({game}, m=${money})")
    ax.set_xlabel("social distance")
    ax.set_ylabel("transfered share")
    ax.set_ylim(-0.02, 0.6)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

    fname = f"{_slug(game)}_{_slug(title)}_m{money}.png"
    return _save(fig, fname)


def main() -> None:
    df_h = human_digitized()
    df_h.to_csv(OUT_DIR / "humans_digitized.csv", index=False)

    df_d = load_llm_means("dictator")
    df_u = load_llm_means("ultimatum")
    df_llm = pd.concat([df_d, df_u], ignore_index=True)

    paths: list[Path] = []

    # Dictator (3 plots)
    paths.append(plot_sonnet_vs_humans(df_llm, df_h, "dictator"))
    paths.append(plot_llms_vs_humans(df_llm, df_h, "dictator", 10))
    paths.append(plot_llms_vs_humans(df_llm, df_h, "dictator", 3000))

    # Ultimatum (3 plots)
    paths.append(plot_sonnet_vs_humans(df_llm, df_h, "ultimatum"))
    paths.append(plot_llms_vs_humans(df_llm, df_h, "ultimatum", 10))
    paths.append(plot_llms_vs_humans(df_llm, df_h, "ultimatum", 3000))

    for p in paths:
        print(p)


if __name__ == "__main__":
    main()


