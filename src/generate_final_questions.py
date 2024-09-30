import pandas as pd
from datasets import load_dataset
import os
import random
import numpy as np


def load_and_process_datasets():
    # Load original and generic-to-brand versions of datasets
    orig_filtered_medqa = load_dataset(
        "AIM-Harvard/gbaker_medqa_usmle_4_options_hf_original", split="test"
    ).to_pandas()
    g2b_medqa = load_dataset(
        "AIM-Harvard/gbaker_medqa_usmle_4_options_hf_generic_to_brand", split="test"
    ).to_pandas()
    orig_filtered_medmcqa = load_dataset(
        "AIM-Harvard/medmcqa_original", split="test"
    ).to_pandas()
    g2b_medmcqa = load_dataset(
        "AIM-Harvard/medmcqa_generic_to_brand", split="test"
    ).to_pandas()

    # Sort datasets
    for df in [orig_filtered_medqa, g2b_medqa, orig_filtered_medmcqa, g2b_medmcqa]:
        df.sort_values("id", inplace=True)

    return orig_filtered_medqa, g2b_medqa, orig_filtered_medmcqa, g2b_medmcqa


def load_annotations():
    # Load annotated datasets
    annotated_medmcqa = pd.read_csv("../data/annotated_medmcqa_new.csv")
    annotated_medqa = pd.read_csv("../data/annotated_medqa_new.csv")

    return annotated_medmcqa, annotated_medqa


def get_rows_to_filter(annotated_df):
    # Convert the keep/drop column to string type
    annotated_df.iloc[:, -2] = annotated_df.iloc[:, -2].astype(str)

    # Get list of ids to filter (where penultimate column is not "keep")
    rows_to_filter = annotated_df[annotated_df.iloc[:, -2] != "keep"].id.tolist()

    return rows_to_filter


def filter_datasets(orig_df, g2b_df, rows_to_filter):
    # Filter out the rows from both original and generic-to-brand datasets
    filtered_orig = orig_df[~orig_df.id.isin(rows_to_filter)]
    filtered_g2b = g2b_df[~g2b_df.id.isin(rows_to_filter)]

    return filtered_orig, filtered_g2b


def save_filtered_datasets(
    filtered_orig_medmcqa, filtered_g2b_medmcqa, filtered_orig_medqa, filtered_g2b_medqa
):
    # Save filtered datasets as parquet files
    filtered_orig_medmcqa.to_parquet("../data/medmcqa/test/original_filtered.parquet")
    filtered_g2b_medmcqa.to_parquet(
        "../data/medmcqa/test/generic_to_brand_filtered.parquet"
    )
    filtered_orig_medqa.to_parquet(
        "../data/GBaker_MedQA-USMLE-4-options-hf/test/original_filtered.parquet"
    )
    filtered_g2b_medqa.to_parquet(
        "../data/GBaker_MedQA-USMLE-4-options-hf/test/generic_to_brand_filtered.parquet"
    )


def main():
    # Load datasets
    orig_filtered_medqa, g2b_medqa, orig_filtered_medmcqa, g2b_medmcqa = (
        load_and_process_datasets()
    )

    # Load annotations
    annotated_medmcqa, annotated_medqa = load_annotations()

    # Get rows to filter
    medmcqa_rows_to_filter = get_rows_to_filter(annotated_medmcqa)
    medqa_rows_to_filter = get_rows_to_filter(annotated_medqa)

    # Filter datasets
    filtered_orig_medmcqa, filtered_g2b_medmcqa = filter_datasets(
        orig_filtered_medmcqa, g2b_medmcqa, medmcqa_rows_to_filter
    )
    filtered_orig_medqa, filtered_g2b_medqa = filter_datasets(
        orig_filtered_medqa, g2b_medqa, medqa_rows_to_filter
    )

    # Print statistics
    print(f"Number of rows filtered in MedMCQA: {len(medmcqa_rows_to_filter)}")
    print(f"Number of rows filtered in MedQA: {len(medqa_rows_to_filter)}")
    print(f"Rows in filtered MedMCQA: {len(filtered_orig_medmcqa)}")
    print(f"Rows in filtered MedQA: {len(filtered_orig_medqa)}")

    # Save filtered datasets
    save_filtered_datasets(
        filtered_orig_medmcqa,
        filtered_g2b_medmcqa,
        filtered_orig_medqa,
        filtered_g2b_medqa,
    )

    print("Filtering complete. Filtered datasets have been saved.")


if __name__ == "__main__":
    main()
