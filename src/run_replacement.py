import os
import pandas as pd
import re
import logging
from typing import Any, Dict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from drug_mapping import DrugMapper

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)


# debugging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Datasets and their columns of interest
DATASETS = {
    "medmcqa": ["question", "opa", "opb", "opc", "opd"],
    "GBaker/MedQA-USMLE-4-options-hf": ["sent1", "ending0", "ending1", "ending3"],
}


def replace_drugs(prompt: str, old_keyword: str, new_keyword: str) -> str:
    """
    Replace occurrences of old_keyword with new_keyword in the given prompt.
    """
    if not isinstance(prompt, str):
        logging.warning(f"Expected a string for replacement, but got {type(prompt)}.")
        return prompt  # Skip non-string types silently

    pattern = re.compile(rf"\b{re.escape(old_keyword)}\b", re.IGNORECASE)
    if pattern.search(prompt):
        logging.debug(f"Found '{old_keyword}' in '{prompt}'")
        replaced_prompt = pattern.sub(new_keyword, prompt)
        if prompt != replaced_prompt:
            logging.debug(
                f"Replaced '{old_keyword}' with '{new_keyword}' in '{prompt}' to get '{replaced_prompt}'"
            )
        return replaced_prompt
    return prompt


def replace_in_col(col_value: Any, replacement_map: Dict[str, str]) -> Any:
    """
    Replace keywords in the column value based on the replacement map.
    """
    if col_value is None or (
        isinstance(col_value, (str, float)) and pd.isna(col_value)
    ):
        return col_value

    if isinstance(col_value, list):
        return [replace_in_col(item, replacement_map) for item in col_value]

    if isinstance(col_value, dict):
        return {
            key: replace_in_col(value, replacement_map)
            for key, value in col_value.items()
        }

    if isinstance(col_value, str):
        logging.debug(f"col_value: {col_value}")
        for old_keyword, new_keyword in replacement_map.items():
            if old_keyword.lower() in col_value.lower():
                logging.debug(f"Found '{old_keyword}' in '{col_value}'")
                col_value = replace_drugs(col_value, old_keyword, new_keyword)
                logging.debug(f"After replacement: '{col_value}'")
    return col_value


def replace_keywords_batch(
    batch_data_df: pd.DataFrame, cols_of_interest: list, replacement_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Replace keywords in the batch DataFrame based on the replacement map.
    """
    for col in cols_of_interest:
        batch_data_df[col] = batch_data_df[col].apply(
            lambda x: replace_in_col(x, replacement_map)
        )

    logging.info("Replacement completed for batch.")
    return batch_data_df


def replace_keywords_parallel(
    split_data_df: pd.DataFrame,
    cols_of_interest: list,
    replacement_map: Dict[str, str],
    max_workers: int = 30,
) -> pd.DataFrame:
    """
    Parallel processing for keyword replacement.
    """
    total_rows = len(split_data_df)
    if total_rows == 0:
        logging.warning("Empty dataset provided.")
        return split_data_df

    batch_size = (total_rows + max_workers - 1) // max_workers
    logging.info(
        f"Processing {total_rows} rows in {max_workers} batches of size {batch_size}"
    )

    futures = []
    num_batches = (total_rows + batch_size - 1) // batch_size

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(
            total=num_batches, desc="Submitting replacement batches", unit="batch"
        ) as submit_progress:
            for start in range(0, total_rows, batch_size):
                end = start + batch_size
                batch_data_df = split_data_df.iloc[start:end]
                futures.append(
                    executor.submit(
                        replace_keywords_batch,
                        batch_data_df,
                        cols_of_interest,
                        replacement_map,
                    )
                )
                submit_progress.update(1)

        final_data = pd.DataFrame()
        with tqdm(
            total=num_batches, desc="Collecting replacement results", unit="batch"
        ) as collect_progress:
            for future in as_completed(futures):
                batch_result = future.result()
                final_data = pd.concat([final_data, batch_result])
                collect_progress.update(1)

    logging.info(f"Total rows after processing: {len(final_data)}")
    return final_data


def process_replacements(
    df: pd.DataFrame,
    brand_to_generic_map: Dict[str, str],
    generic_to_brand_map: Dict[str, str],
    cols_of_interest: list,
) -> tuple:
    """
    Process replacements for the given dataframe and maps.
    """
    logging.info("Processing replacements...")

    original_df = df.copy()
    brand_to_generic_df = replace_keywords_parallel(
        df.copy(), cols_of_interest, brand_to_generic_map
    )
    generic_to_brand_df = replace_keywords_parallel(
        df.copy(), cols_of_interest, generic_to_brand_map
    )

    return original_df, brand_to_generic_df, generic_to_brand_df


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    mapper = DrugMapper(args.brand_to_generic_csv_path, args.generic_to_brand_csv_path)
    brand_to_generic_map = mapper.load_keywords("brand_to_generic")
    logging.debug(f"brand_to_generic_map: {brand_to_generic_map}")
    generic_to_brand_map = mapper.load_keywords("generic_to_brand")
    logging.debug(f"generic_to_brand_map: {generic_to_brand_map}")
    all_keywords = mapper.load_all_keywords_list()
    logging.debug(f"All keywords: {all_keywords}")

    for dataset_name, cols_of_interest in DATASETS.items():
        cleaned_dataset_name = dataset_name.replace("/", "_")
        dataset_output_dir = os.path.join(args.input_dir, cleaned_dataset_name)
        logging.info(f"output_dir: {dataset_output_dir}")
        if not os.path.exists(dataset_output_dir):
            logging.warning(f"Directory for dataset {dataset_name} not found.")
            continue

        for split in ["train", "dev", "validation", "test"]:
            split_dir = os.path.join(dataset_output_dir, split)
            if not os.path.exists(split_dir):
                logging.warning(f"Directory for split {split} not found.")
                logging.info(f"No valid dir at {split_dir}")
                continue

            filtered_file_path = os.path.join(
                split_dir, f"{cleaned_dataset_name}_{split}_filtered.parquet"
            )
            if not os.path.exists(filtered_file_path):
                logging.warning(f"Filtered file for {split} split not found.")
                logging.info(f"No valid file at {filtered_file_path}")
                continue

            logging.info(f"Processing {filtered_file_path}...")
            df = pd.read_parquet(filtered_file_path)

            original_df, brand_to_generic_df, generic_to_brand_df = (
                process_replacements(
                    df, brand_to_generic_map, generic_to_brand_map, cols_of_interest
                )
            )

            original_output_path = os.path.join(split_dir, "original.parquet")
            brand_to_generic_output_path = os.path.join(
                split_dir, "brand_to_generic.parquet"
            )
            generic_to_brand_output_path = os.path.join(
                # Save the results
                split_dir,
                "generic_to_brand.parquet",
            )

            original_df.to_parquet(original_output_path, index=False)
            brand_to_generic_df.to_parquet(brand_to_generic_output_path, index=False)
            generic_to_brand_df.to_parquet(generic_to_brand_output_path, index=False)

            logging.info(f"Saved original data to: {original_output_path}")
            logging.info(
                f"Saved brand-to-generic data to: {brand_to_generic_output_path}"
            )
            logging.info(
                f"Saved generic-to-brand data to: {generic_to_brand_output_path}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process replacements in filtered datasets."
    )
    parser.add_argument(
        "--brand_to_generic_csv_path",
        type=str,
        default="../data/generic_to_brand.csv",
        help="Path to the CSV file containing brand to generic drug mappings.",
    )
    parser.add_argument(
        "--generic_to_brand_csv_path",
        type=str,
        default="../data/generic_to_brand.csv",
        help="Path to the CSV file containing generic to brand drug mappings.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="pre_filter_datasets",
        help="Directory containing the filtered Parquet files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_datasets",
        help="Directory to save the processed datasets.",
    )
    args = parser.parse_args()

    main(args)
