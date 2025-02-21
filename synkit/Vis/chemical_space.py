import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rc("text", usetex=True)  # Enable LaTeX rendering
plt.rc("font", family="serif")  # Optional: use serif font


def scatter_plot(
    data_train,
    data_test,
    size_train=10,
    size_test=10,
    title=None,
    ax=None,
    xlabel="Coordinate 1",
    ylabel="Coordinate 2",
):
    # Check if data is empty
    if data_train.empty or data_test.empty:
        raise ValueError("Input data frames cannot be empty.")

    # Check for necessary columns
    if data_train.columns.size < 3 or data_test.columns.size < 3:
        raise ValueError("Data frames must have at least three columns.")

    # Adding 'Type' column to differentiate between train and test data
    data_train["Type"] = "Train"
    data_test["Type"] = "Test"

    # Combine the datasets
    data_combined = pd.concat([data_train, data_test])

    # If no axes object is passed, create one
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Define a more distinct color palette
    pastel_palette = {
        "Train": "deepskyblue",
        "Test": "magenta",
    }  # Using deepskyblue and magenta for better distinction

    # Create scatter plots with specified sizes
    for dtype, color in pastel_palette.items():
        subset = data_combined[data_combined["Type"] == dtype]
        ax.scatter(
            subset[subset.columns[1]],
            subset[subset.columns[2]],
            color=color,
            label=dtype,
            s=size_train if dtype == "Train" else size_test,
            alpha=0.1,
            edgecolor="none",
        )

    # Set the title if provided
    if title:
        ax.set_title(rf"{title}", fontsize=24, fontweight="bold")

    # Set labels
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)

    # Enhance grid and layout
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Get legend handles and labels for external usage
    handles, labels = ax.get_legend_handles_labels()

    # Return the axes, handles, and labels for further customization outside the function
    return ax, handles, labels


# Define a function that modifies the legend handles to full opacity for better visibility in the legend
def adjust_legend_handles(handles, colors):
    new_handles = []
    for handle, color in zip(handles, colors):
        # Create a new handle with the same properties but with full alpha for the legend
        new_handle = mpatches.Patch(color=color, label=handle.get_label())
        new_handles.append(new_handle)
    return new_handles
