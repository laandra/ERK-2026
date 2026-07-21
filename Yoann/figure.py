"""
Section 9 -- Cross-dataset comparison figures (ERK paper, Section 3.1)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def make_paper_figures(summary_df: pd.DataFrame = None,
                        output_root: str = "results") -> pd.DataFrame:
    if summary_df is None:
        summary_df = pd.read_csv(
            os.path.join(output_root, "summary_all_datasets.csv"),
            index_col="dataset",
        )

    df = summary_df.copy()

    bad = df["gain_oracle_vs_no_battery_pct"] <= 0
    if bad.any():
        print("WARNING -- non-positive oracle gain for:", df.index[bad].tolist())

    df["fraction_theoretical_gain_pct"] = 100 * (
        df["gain_prophet_vs_no_battery_pct"] / df["gain_oracle_vs_no_battery_pct"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(7, 4.5))
    for ax, col, ylabel in [
        (axes[0], "fraction_theoretical_gain_pct", "Fraction of theoretical gain (%)"),
        (axes[1], "gain_prophet_vs_no_battery_pct", "Cost improvement vs. no battery (%)"),
    ]:
        data = df[col].dropna().values
        ax.boxplot(data, widths=0.5)
        x_jitter = 1 + np.random.uniform(-0.05, 0.05, size=len(data))
        ax.scatter(x_jitter, data, alpha=0.5, s=14, color="steelblue")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1])
        ax.set_xticklabels(["MILP+Prophet"])
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_root, "fig_milp_prophet_boxplots.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    print("\n--- Stats for the paper text ---")
    for col, label in [
        ("gain_prophet_vs_no_battery_pct", "Cost improvement vs. no battery (%)"),
        ("fraction_theoretical_gain_pct", "Fraction of theoretical gain (%)"),
        ("regret_prophet_usd", "Regret vs. oracle (USD)"),
    ]:
        s = df[col].dropna()
        print(f"{label}:")
        print(f"   median = {s.median():.1f}   "
              f"min = {s.min():.1f}   max = {s.max():.1f}   "
              f"Q1 = {s.quantile(0.25):.1f}   Q3 = {s.quantile(0.75):.1f}")

    median_val = df["fraction_theoretical_gain_pct"].median()
    df["_dist_to_median"] = (df["fraction_theoretical_gain_pct"] - median_val).abs()
    median_dataset = df["_dist_to_median"].idxmin()
    df.drop(columns="_dist_to_median", inplace=True)

    illustrative_fig = os.path.join(
        output_root, median_dataset, f"hems_scenarios_{median_dataset}.png"
    )
    print(f"\nRepresentative (median) household: {median_dataset}")
    print(f"-> reuse this figure directly for Figure X+2: {illustrative_fig}")

    df.to_csv(os.path.join(output_root, "summary_with_fraction.csv"), encoding="utf-8-sig")
    return df


def plot_scenarios_fixed(df_fc, df_pk, delta_t=0.5, save_path=None, title_suffix=""):
    """
    Version corrigée de plot_scenarios() : les annotations flottantes
    qui se chevauchaient ont été retirées. L'encart "Summary" en haut
    à gauche donne déjà les 3 chiffres (Oracle / Prophet / Regret).
    """
    buy_nb  = np.maximum(0, df_fc["Consumption"] - df_fc["Solar_Gen"])
    sell_nb = np.maximum(0, df_fc["Solar_Gen"]   - df_fc["Consumption"])

    cum_nb = ((buy_nb * df_fc["Buy_Rate_USD_kWh"] - sell_nb * df_fc["Sell_Rate_USD_kWh"]) * delta_t).cumsum()
    cum_fc = df_fc["Step_Cost_USD"].cumsum()
    cum_pk = df_pk["Step_Cost_USD"].cumsum()

    gain_pk = cum_nb - cum_pk
    gain_fc = cum_nb - cum_fc
    regret  = cum_fc - cum_pk

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.fill_between(df_fc.index, cum_pk, cum_fc, alpha=0.25, color="orange", label="Regret")
    ax.fill_between(df_fc.index, cum_fc, cum_nb, alpha=0.15, color="steelblue", label="Prophet gain")

    ax.plot(df_fc.index, cum_nb, color="tomato", lw=2, label="No battery")
    ax.plot(df_pk.index, cum_pk, color="darkgreen", lw=2, ls="--", label="Perfect foresight")
    ax.plot(df_fc.index, cum_fc, color="steelblue", lw=2, label="Prophet forecast")

    summary = (f"""Summary
Oracle: {gain_pk.iloc[-1]:.2f} USD
Prophet: {gain_fc.iloc[-1]:.2f} USD
Regret: {regret.iloc[-1]:.2f} USD""")
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    ax.set_ylabel("Cumulative cost (USD)")
    ax.set_title(f"Scenario comparison {title_suffix}".strip())
    ax.grid(alpha=0.3)
    ax.legend()

    n_days = (df_fc.index[-1] - df_fc.index[0]).days + 1
    if n_days <= 3:
        locator, fmt = mdates.HourLocator(interval=6), "%d/%m %Hh"
    elif n_days <= 14:
        locator, fmt = mdates.DayLocator(interval=1), "%d/%m"
    elif n_days <= 60:
        locator, fmt = mdates.DayLocator(interval=5), "%d/%m"
    elif n_days <= 180:
        locator, fmt = mdates.DayLocator(interval=15), "%d/%m/%y"
    else:
        locator, fmt = mdates.MonthLocator(interval=1), "%m/%Y"

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fix_illustrative_figure(dataset_name: str, output_root: str = "results"):
    """
    Recharge les CSV déjà sauvegardés pour `dataset_name` et régénère
    la figure Figure X+2 avec la version corrigée de plot_scenarios().
    """
    out_dir = os.path.join(output_root, dataset_name)

    df_fc = pd.read_csv(
        os.path.join(out_dir, f"df_fc_{dataset_name}.csv"),
        index_col="Timestamp", parse_dates=True,
    )
    df_pk = pd.read_csv(
        os.path.join(out_dir, f"df_pk_{dataset_name}.csv"),
        index_col="Timestamp", parse_dates=True,
    )

    save_path = os.path.join(out_dir, f"hems_scenarios_{dataset_name}.png")
    plot_scenarios_fixed(df_fc, df_pk, delta_t=0.5, save_path=save_path, title_suffix=f"— {dataset_name}")
    print(f"Figure régénérée : {save_path}")


if __name__ == "__main__":
    make_paper_figures(output_root="results")
    fix_illustrative_figure("Ausgrid 65", output_root="results")