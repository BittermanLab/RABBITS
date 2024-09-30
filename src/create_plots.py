import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_b4bqa_bar_graph(data):
    # Drop rows with missing values
    data_clean = data.dropna()

    # Sort data by model size
    sorted_data = data_clean.sort_values(by="size")

    # Extract the relevant columns
    models = sorted_data["Model"]
    sizes = sorted_data["size"]
    b4bqa_scores = sorted_data["b4bqa"]

    # Create the bar plot using seaborn for better coloring
    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=b4bqa_scores, palette=pal)

    # Adding grid
    plt.grid(axis="y", linestyle="--", linewidth=0.7)

    # Labeling the axes
    plt.xlabel("Model (ranked by model size)")
    plt.ylabel("Accuracy")

    # Adding title
    plt.title("Testing Drug's Brand and Generic Terms Matching Performance")

    # Show the plot
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    # plt.show()


def plot_avg_orig_vs_avg_g2b_with_larger_fonts(data):

    # Rename columns to match the function expectations
    data = data.rename(columns={"Original": "avg_orig", "g2b": "avg_g2b"})

    # Drop rows with missing values in avg_orig or avg_g2b
    data_clean = data.dropna(subset=["avg_orig", "avg_g2b"])

    # Define a diverse color palette and ensure it has enough colors
    unique_models = data_clean["Model"].unique()
    palette = sns.color_palette("husl", len(unique_models))

    # Define marker styles for different groups
    model_shapes = {
        "Llama": "o",
        "Qwen": "X",
        "GPT": "D",
        "Phi": "^",
        "Mistral": "v",
        "Mixtral": "<",
        "claude": ">",
        "Gemini": "p",
        "aya": "*",
        "command": "H",
        "Yi": "X",
        "default": "h",
    }

    # Function to assign markers based on model name
    def get_marker(model_name):
        for key in model_shapes:
            if key in model_name:
                return model_shapes[key]
        return model_shapes["default"]

    # Create a column for markers in the dataframe
    data_clean["Marker"] = data_clean["Model"].apply(get_marker)

    # Map colors to models
    model_to_color = {model: palette[i] for i, model in enumerate(unique_models)}

    # Create the scatter plot
    plt.figure(figsize=(12, 7))
    for model in unique_models:
        subset = data_clean[data_clean["Model"] == model]
        sns.scatterplot(
            x="avg_orig",
            y="avg_g2b",
            hue="Model",
            style="Model",
            s=150,  # Increase marker size
            palette=[model_to_color[model]],
            markers=[subset["Marker"].values[0]],
            data=subset,
            legend=False,
        )

    # Adding grid and diagonal line
    plt.plot(
        [data_clean["avg_orig"].min(), data_clean["avg_g2b"].max()],
        [data_clean["avg_orig"].min(), data_clean["avg_g2b"].max()],
        "k--",
    )

    # Labeling the axes with larger font size
    plt.xlabel("Acc with No Replacement", fontsize=14)
    plt.ylabel("Acc with Generic to Brand Replacement", fontsize=14)

    # Adding title with larger font size
    plt.title("Acc of Original vs Generic to Brand Replacement on RABBITS", fontsize=16)

    # Custom legend creation
    handles = []
    labels = []
    for model in unique_models:
        row = data_clean[data_clean["Model"] == model].iloc[0]
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=row["Marker"],
                color="w",
                label=row["Model"],
                markerfacecolor=model_to_color[model],
                markersize=10,
            )
        )
        labels.append(
            f"{row['Model']} | orig: {row['avg_orig']:.2f}, g2b: {row['avg_g2b']:.2f}, diff: {row['Difference']:.2f}"
        )

    plt.legend(
        handles,
        labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=10,  # Reduced font size to fit longer legend
        title_fontsize="12",
    )

    # Adjusting the background shading to be less obtrusive
    plt.axhspan(20, 40, facecolor="lightgrey", alpha=0.4)
    plt.axhspan(40, 60, facecolor="lightblue", alpha=0.4)
    plt.axhspan(60, 80, facecolor="lightgreen", alpha=0.4)
    plt.axhspan(80, 100, facecolor="lightyellow", alpha=0.4)

    # Show the plot with larger tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(20, 100)
    plt.tight_layout()
    plt.savefig("rabbits_plot.png", dpi=300)


def plot_side_by_side_medqa_medmcqa_diff(data):
    # Drop rows with missing values in medmcqa_diff or medqa_diff
    data_clean = data.dropna(subset=["medmcqa_diff", "medqa_diff"])

    # Sort data by model size
    sorted_data = data_clean.sort_values(by="size")

    # Create figure and subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8), sharey=True)

    # Plot for medmcqa_diff
    sns.barplot(x="medmcqa_diff", y="Model", data=sorted_data, ax=axes[0], palette=pal)
    axes[0].set_title("Impact of Generic2Brand swap on MedMCQA Accuracy", fontsize=14)
    axes[0].set_xlabel("MedMCQA acc differ", fontsize=12)
    axes[0].set_ylabel("Model (ranked by model size)", fontsize=12)
    axes[0].tick_params(axis="y", labelsize=10)
    axes[0].tick_params(axis="x", labelsize=12)

    # Plot for medqa_diff
    sns.barplot(
        x="medqa_diff",
        y="Model",
        data=sorted_data,
        ax=axes[1],
        palette=pal,  #'RdYlGn_r'
    )
    axes[1].set_title("Impact of Generic2Brand swap on MedQA Accuracy", fontsize=14)
    axes[1].set_xlabel("MedQA acc differ", fontsize=12)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", labelsize=12)

    # Adjust layout
    plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    data = pd.read_csv("results/rabbit_results.csv")

    # rename model row phi-1_5 to phi1.5
    data["Model"] = data["Model"].replace("phi-1_5", "phi1.5")
    data["Model"] = data["Model"].replace("Phi-3-medium-4k", "phi-3-medium-4k")

    pal = sns.color_palette("viridis", len(data["Model"]))  # viridis

    # Scaling law
    # plot_b4bqa_bar_graph(data)
    # plt.savefig("b4bqa.png", dpi=300)

    # Main comparison plot
    plot_avg_orig_vs_avg_g2b_with_larger_fonts(data)

    # # Side by side dataset
    # plot_side_by_side_medqa_medmcqa_diff(data)
    # plt.savefig("accuracy_difference_generic_to_brand.png", dpi=300)

    # # generate latex table
    # new_table_data = []
    # column_mapping = {"medmcqa_g2b": "g2b", "medmcqa_orig_filtered": "original"}

    # for _, row in data.iterrows():
    #     new_table_data.append(
    #         {
    #             "Dataset": "medmcqa",
    #             "Model": row["Model"],
    #             "g2b": row["medmcqa_g2b"],
    #             "original": row["medmcqa_orig_filtered"],
    #         }
    #     )
    #     new_table_data.append(
    #         {
    #             "Dataset": "medqa 4options",
    #             "Model": row["Model"],
    #             "g2b": row["medqa_4options_g2b"],
    #             "original": row["medqa_4options_orig_filtered"],
    #         }
    #     )

    # new_table_df = pd.DataFrame(new_table_data)

    # # Sorting the dataframe for better readability
    # new_table_df = new_table_df.sort_values(by=["Dataset", "Model"])

    # # Print the dataframe in LaTeX tabular format
    # latex_table = new_table_df.to_latex(index=False, float_format="%.2f")
    # latex_table

    # ## Summary version
    # # Creating the new summary table dataframe
    # summary_table_data = []

    # for _, row in data.iterrows():
    #     if (
    #         pd.notna(row["avg_orig"])
    #         and pd.notna(row["avg_g2b"])
    #         and pd.notna(row["avg_diff"])
    #     ):
    #         summary_table_data.append(
    #             {
    #                 "Model": row["Model"],
    #                 "Original": row["avg_orig"],
    #                 "g2b": row["avg_g2b"],
    #                 "Average": (row["avg_orig"] + row["avg_g2b"]) / 2,
    #                 "Difference": row["avg_diff"],
    #             }
    #         )

    # summary_table_df = pd.DataFrame(summary_table_data)

    # # Sorting the dataframe for better readability
    # summary_table_df = summary_table_df.sort_values(by=["Model"])

    # # Print the dataframe in LaTeX tabular format
    # summary_latex_table = summary_table_df.to_latex(index=False, float_format="%.2f")

    # print(summary_latex_table)
