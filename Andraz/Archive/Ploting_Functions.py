### CELL 13: Function for Plotting Graphs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, datetime

"Funkcija za risanje grafov"
def plotMultiY(X, Y=None, X_label="X_os", Y_Label=None,
               legend=None, title="title",
               save=False, grid=True, save_pdf=False, show_title=False):

    if Y is None:
        Y = []
    if Y_Label is None:
        Y_Label = ["" for _ in range(max(1, len(Y)))]
    if legend is None:
        legend = ["" for _ in range(max(1, len(Y)))]

    # Normalize to mutable Python lists; callers often pass numpy/pandas objects.
    X = list(X)
    Y = [list(series) for series in Y]
    Y_Label = list(Y_Label)
    legend = list(legend)

    barve = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    line_type = ["-", "--", ":", "-."]

    if show_title:
        plt.title(title)

    # Popravek dolžin Y_Label in legend
    if len(Y_Label) < len(Y):
        for _ in range(len(Y) - len(Y_Label)):
            Y_Label.append(Y_Label[0])

    if len(legend) < len(Y):
        for _ in range(len(Y) - len(legend)):
            legend.append(legend[0])

    # Popravek dolžin X in Y
    if len(Y) > 0:
        if len(X) < len(Y[0]):
            X += [X[-1]] * (len(Y[0]) - len(X))
        elif len(X) > len(Y[0]):
            X = X[:len(Y[0])]

    for i in range(len(Y) - 1):
        if len(Y[i + 1]) < len(Y[i]):
            Y[i + 1] += [Y[i + 1][-1]] * (len(Y[i]) - len(Y[i + 1]))
        elif len(Y[i + 1]) > len(Y[i]):
            Y[i + 1] = Y[i + 1][:len(Y[i])]

    # Mreža
    if grid:
        plt.grid(color="Black", linestyle="--", linewidth=0.5)
    else:
        plt.grid(False)

    # Risanje grafov
    for i in range(len(Y)):
        plt.plot(X, Y[i], color=barve[i], label=legend[i], linestyle=line_type[i % len(line_type)])
        plt.legend()
        if i > 0:
            if Y_Label[i] != Y_Label[i - 1]:
                plt.ylabel(Y_Label[i], color=barve[i])
                plt.tick_params(axis="y", colors=barve[i])
        else:
            plt.xlabel(X_label)
            plt.ylabel(Y_Label[i] if len(Y_Label) > 0 else "")

    # Preveri, ali so X datumi
    if all(isinstance(x, (date, datetime)) for x in X):
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        # Če ni datumska os, prikaži vsako n-to oznako
        if len(X) > 10:
            step = max(1, len(X) // 10)  # max 10 oznak
            plt.xticks(X[::step], rotation=45)

    # Adjust layout to prevent labels from overlapping
    plt.tight_layout()

    # Shranjevanje
    if save:
        plt.savefig(f"{title}.png")
    if save_pdf:
        plt.savefig(f"{title}.pdf")

    plt.show()


# ============================================================================
# BTR (Beyond the Rainbow) Plotting Functions
# ============================================================================

def plot_btr_comparison(comparison_df, metric="price_mean", title="BTR Algorithm Comparison", save=False):
    """
    Plot comparison of BTR algorithms and/or SB3 algorithms.
    
    Args:
        comparison_df: DataFrame with columns: algorithm, metric_mean, metric_std
        metric: Metric to plot ("price_mean", "reward_mean", etc.)
        title: Plot title
        save: If True, save plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Prepare data
    algorithms = comparison_df["algorithm"].tolist()
    values = comparison_df[metric].tolist()
    errors = comparison_df.get(metric.replace("_mean", "_std"), [0] * len(algorithms)).tolist()
    
    # Color coding
    colors = []
    for algo in algorithms:
        algo_upper = str(algo).upper()
        if "MILP" in algo_upper:
            colors.append("#111111")
        elif "IQN" in algo_upper or "C51" in algo_upper:
            colors.append("#1f77b4")
        else:
            colors.append("#ff7f0e")
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(algorithms))
    bars = ax.bar(x_pos, values, yerr=errors, capsize=5, color=colors, alpha=0.7, edgecolor="black")
    
    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    
    plt.show()


def plot_training_progress(timestamps, rewards, prices=None, algorithm_name="BTR", title="Training Progress", save=False):
    """
    Plot training progress (reward and price evolution).
    
    Args:
        timestamps: List of timesteps during training
        rewards: List of evaluation rewards
        prices: List of energy costs (optional)
        algorithm_name: Algorithm name for legend
        title: Plot title
        save: If True, save plot
    """
    import matplotlib.pyplot as plt
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot rewards
    color = "tab:blue"
    ax1.set_xlabel("Training Timesteps", fontsize=12)
    ax1.set_ylabel("Evaluation Reward", color=color, fontsize=12)
    ax1.plot(timestamps, rewards, color=color, marker="o", label=f"{algorithm_name} Reward", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(alpha=0.3)
    
    # Plot prices on secondary y-axis
    if prices is not None:
        ax2 = ax1.twinx()
        color = "tab:orange"
        ax2.set_ylabel("Energy Cost (EUR)", color=color, fontsize=12)
        ax2.plot(timestamps, prices, color=color, marker="s", label=f"{algorithm_name} Cost", linewidth=2)
        ax2.tick_params(axis="y", labelcolor=color)
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    else:
        ax1.legend(fontsize=10)
    
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save:
        plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    
    plt.show()


def plot_price_distribution(results_df, algorithm_filter=None, title="Energy Cost Distribution", save=False):
    """
    Plot distribution of energy costs across seeds using violin/box plots.
    
    Args:
        results_df: DataFrame with columns: algorithm, price_mean, seed
        algorithm_filter: List of algorithms to include (None = all)
        title: Plot title
        save: If True, save plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if algorithm_filter:
        df = results_df[results_df["algorithm"].isin(algorithm_filter)]
    else:
        df = results_df
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create violin plot
    sns.violinplot(data=df, x="algorithm", y="price_mean", ax=ax, palette="Set2")
    sns.boxplot(data=df, x="algorithm", y="price_mean", ax=ax, width=0.3, palette="Set2")
    
    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_ylabel("Energy Cost (EUR)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    
    plt.show()


def plot_reward_vs_price(comparison_df, title="Reward vs Energy Cost Trade-off", save=False):
    """
    Plot scatter: reward vs energy cost to visualize trade-offs.
    
    Args:
        comparison_df: DataFrame with reward_mean, price_mean, algorithm columns
        title: Plot title
        save: If True, save plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = comparison_df["algorithm"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    
    for algo, color in zip(algorithms, colors):
        algo_data = comparison_df[comparison_df["algorithm"] == algo]
        ax.scatter(algo_data["price_mean"], algo_data["reward_mean"],
                   s=200, alpha=0.6, label=algo, color=color, edgecolors="black", linewidth=2)
        
        # Add algorithm labels
        for idx, row in algo_data.iterrows():
            ax.annotate(algo, (row["price_mean"], row["reward_mean"]),
                       textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    
    ax.set_xlabel("Energy Cost (EUR)", fontsize=12)
    ax.set_ylabel("Evaluation Reward", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    
    plt.show()


def plot_algorithm_ranking(comparison_df, metric="price_mean", ascending=True, title="Algorithm Ranking", save=False):
    """
    Plot algorithms ranked by a metric.
    
    Args:
        comparison_df: DataFrame with algorithm and metric columns
        metric: Metric to rank by
        ascending: If True, lower values rank higher
        title: Plot title
        save: If True, save plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Sort data
    df_sorted = comparison_df.sort_values(metric, ascending=ascending).reset_index(drop=True)
    df_sorted["rank"] = range(1, len(df_sorted) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(df_sorted))
    colors = []
    gradient = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_sorted)))
    for idx, algo in enumerate(df_sorted["algorithm"].astype(str).values):
        if "MILP" in algo.upper():
            colors.append("#111111")
        else:
            colors.append(gradient[idx])
    
    bars = ax.barh(x_pos, df_sorted[metric].values, color=colors, edgecolor="black", linewidth=1.5)
    
    ax.set_yticks(x_pos)
    ax.set_yticklabels(df_sorted["algorithm"].values, fontsize=11)
    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    
    # Add value labels and rank
    for i, (bar, val, rank) in enumerate(zip(bars, df_sorted[metric].values, df_sorted["rank"].values)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {val:.2f}',
                ha='left', va='center', fontsize=10, fontweight="bold")
        ax.text(0.01, bar.get_y() + bar.get_height()/2.,
                f'#{int(rank)}',
                ha='left', va='center', fontsize=10, color="white", fontweight="bold")
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    
    plt.show()