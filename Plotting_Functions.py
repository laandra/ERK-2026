"""Shared multi-series line plot used by the notebooks."""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, datetime


def plotMultiY(X, Y=None, X_label="X_os", Y_Label=None,
               legend=None, title="title",
               save=False, grid=True, save_pdf=False, show_title=False):
    """Plot several Y series against a shared X axis, padding mismatched lengths."""

    if Y is None:
        Y = []
    if Y_Label is None:
        Y_Label = ["" for _ in range(max(1, len(Y)))]
    if legend is None:
        legend = ["" for _ in range(max(1, len(Y)))]

    X = list(X)
    Y = [list(series) for series in Y]
    Y_Label = list(Y_Label)
    legend = list(legend)

    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    line_type = ["-", "--", ":", "-."]

    if show_title:
        plt.title(title)

    if len(Y_Label) < len(Y):
        for _ in range(len(Y) - len(Y_Label)):
            Y_Label.append(Y_Label[0])

    if len(legend) < len(Y):
        for _ in range(len(Y) - len(legend)):
            legend.append(legend[0])

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

    if grid:
        plt.grid(color="Black", linestyle="--", linewidth=0.5)
    else:
        plt.grid(False)

    for i in range(len(Y)):
        plt.plot(X, Y[i], color=colors[i], label=legend[i], linestyle=line_type[i % len(line_type)])
        plt.legend()
        if i > 0:
            if Y_Label[i] != Y_Label[i - 1]:
                plt.ylabel(Y_Label[i], color=colors[i])
                plt.tick_params(axis="y", colors=colors[i])
        else:
            plt.xlabel(X_label)
            plt.ylabel(Y_Label[i] if len(Y_Label) > 0 else "")

    # Date axis gets month ticks; otherwise ~10 evenly spaced labels.
    if all(isinstance(x, (date, datetime)) for x in X):
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        if len(X) > 10:
            step = max(1, len(X) // 10)
            plt.xticks(X[::step], rotation=45)

    plt.tight_layout()

    if save:
        plt.savefig(f"{title}.png")
    if save_pdf:
        plt.savefig(f"{title}.pdf")

    plt.show()
