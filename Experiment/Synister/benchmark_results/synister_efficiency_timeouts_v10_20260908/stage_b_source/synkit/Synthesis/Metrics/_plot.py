import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Enable LaTeX rendering in Matplotlib
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def plot_recognition_coverage_curve(
    data,
    coverage_col="Coverage",
    recognition_col="Recognition",
    f2_score_col="F2_score",
    figsize=(8, 6),
    show_f2=True,
    show_legend=True,
):
    """Plots a Recognition-Coverage curve using provided data, including
    optional F2 scores annotated. Styled with Seaborn for enhanced visual
    appearance.

    :param data: Nested dictionary containing the data for each radii,
                 formatted as shown in example.
    :type data: dict
    :param coverage_col: Key name for the coverage data in the dictionary.
    :type coverage_col: str
    :param recognition_col: Key name for the recognition data in the dictionary.
    :type recognition_col: str
    :param f2_score_col: Key name for the F2 score data in the dictionary.
    :type f2_score_col: str
    :param figsize: Figure size for the plot, default is (8, 6).
    :type figsize: tuple
    :param show_f2: Whether to show F2 scores on the curve, default is True.

                    Example Data format:
                    {'radii_0': {'Novelty': 96.44, 'Coverage': 93.98, 'Recognition': 3.55, ...}}
    :type show_f2: bool
    """
    df = pd.DataFrame(data).T

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=figsize)
    ax = sns.lineplot(
        data=df,
        x=coverage_col,
        y=recognition_col,
        marker="o",
        label="Recognition-Coverage Curve" if show_legend else None,
    )

    if show_f2:
        for i, txt in enumerate(df[f2_score_col]):
            ax.annotate(
                r"$F_2=" + f"{txt:.2f}" + "$",
                (df[coverage_col].iloc[i], df[recognition_col].iloc[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

    # Adding labels, title, and legend with LaTeX
    plt.xlabel(r"$\mathrm{{Coverage\%}}$", fontsize=18)
    plt.ylabel(r"$\mathrm{{Recognition\%}}$", fontsize=18)
    plt.title(r"Recognition-Coverage Curve", fontsize=24)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_f2_scores_line(data, figsize=(8, 6), show_f2=True, show_legend=True):
    """Plots F2 scores across different radii using a line plot, showing the
    trend of F2 score changes, and annotated with optional F2 scores.

    :param data: Dictionary containing nested dictionaries with 'F2_score'
                 and possibly other metrics.
    :type data: dict
    :param figsize: Figure size for the plot, default is (8, 6).
    :type figsize: tuple
    :param show_f2: Whether to show F2 scores on the curve, default is True.
    :type show_f2: bool
    :param show_legend: Whether to show the legend on the plot, default is True.

                        Example Data format:
                        {'radii_0': {'Novelty': 96.44, 'Coverage': 93.98, 'Recognition': 3.55,
                        'F2_score': 0.15}, ...}
    :type show_legend: bool
    """
    # Convert the nested dictionary into a DataFrame and prepare for plotting
    df = pd.DataFrame(data).T
    df["Radii"] = [int(key.split("_")[1]) for key in data.keys()]
    df.sort_values("Radii", inplace=True)  # Ensure data is sorted by radii

    # Setting up Seaborn for enhanced plotting style
    sns.set_theme(style="whitegrid")

    # Create the plot
    plt.figure(figsize=figsize)
    sns.lineplot(
        data=df,
        x="Radii",
        y="F2_score",
        marker="o",
        label="F2 Score Trend" if show_legend else None,
    )

    # Optionally annotate each point with its F2 score
    if show_f2:
        for i in range(len(df)):
            plt.text(
                df.iloc[i]["Radii"],
                df.iloc[i]["F2_score"] + 0.01,
                r"$F_2=" + f'{df.iloc[i]["F2_score"]:.2f}' + "$",
                color="black",
                ha="center",
                va="bottom",
            )

    # Adding labels, title, and legend with LaTeX
    plt.xlabel(r"$\text{Radii}$", fontsize=18)
    plt.ylabel(r"$F_2\ \text{Score}$", fontsize=18)
    plt.title(r"$F_2\ \text{Score}$ Across Different Radii", fontsize=24)
    if show_legend:
        plt.legend()

    plt.grid(True)
    plt.show()
