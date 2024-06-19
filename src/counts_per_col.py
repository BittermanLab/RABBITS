import os
import pandas as pd
import re
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import collections
import argparse
from datasets import load_dataset
from drug_mapping import DrugMapper

# Adjust pandas display options
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.max_colwidth", None)  # Show full column content
pd.set_option("display.width", 1000)  # Adjust the display width

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_keywords(col_value, keywords):
    found_keywords = []
    if col_value is None or (
        isinstance(col_value, (str, float)) and pd.isna(col_value)
    ):
        return found_keywords

    keywords = sorted(keywords, key=len, reverse=True)

    if isinstance(col_value, list):
        for item in col_value:
            for keyword in keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", item, re.IGNORECASE):
                    found_keywords.append(keyword)

    if isinstance(col_value, str):
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", col_value, re.IGNORECASE):
                found_keywords.append(keyword)

    logging.debug(f"Extracted keywords from '{col_value}': {found_keywords}")
    return list(set(found_keywords))


def extract_keywords_batch(batch_data_df, cols_of_interest, keywords):
    logging.info(f"Extracting keywords for batch with {len(batch_data_df)} rows.")

    batch_data_df["found_keywords"] = batch_data_df[cols_of_interest].apply(
        lambda row: list(
            set(keyword for cell in row for keyword in extract_keywords(cell, keywords))
        ),
        axis=1,
    )

    logging.info(
        f"Keywords extracted for batch. Found keywords: {batch_data_df['found_keywords'].apply(len).sum()}"
    )
    return batch_data_df


def modify_dataset_parallel(
    split_data_df, cols_of_interest, replacement_map, keywords, max_workers=4
):
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
            total=num_batches,
            desc="Submitting keyword extraction batches",
            unit="batch",
        ) as submit_progress:
            for start in range(0, total_rows, batch_size):
                end = start + batch_size
                batch_data_df = split_data_df.iloc[start:end]
                futures.append(
                    executor.submit(
                        extract_keywords_batch,
                        batch_data_df,
                        cols_of_interest,
                        keywords,
                    )
                )
                submit_progress.update(1)

        modified_data = pd.DataFrame()
        with tqdm(
            total=num_batches,
            desc="Collecting keyword extraction results",
            unit="batch",
        ) as collect_progress:
            for future in as_completed(futures):
                batch_result = future.result()
                logging.info(f"Batch processed with {len(batch_result)} rows.")
                modified_data = pd.concat([modified_data, batch_result])
                collect_progress.update(1)

    return modified_data


def process_split_in_chunks(
    dataset_split,
    drug_mapper,
    brand_to_generic_map,
    generic_to_brand_map,
    cols_of_interest,
    output_folder,
    dataset_name,
    split_name,
    max_workers=4,
):
    dataset_split_df = dataset_split.to_pandas()

    # Create unique id for each row
    dataset_split_df["local_id"] = dataset_split_df.index

    keywords = drug_mapper.load_all_keywords_list()

    # Extract keywords and filter rows with no keywords found
    extracted_data = modify_dataset_parallel(
        dataset_split_df,
        cols_of_interest,
        None,  # No replacement map for extraction
        keywords,
        max_workers=max_workers,
    )

    filtered_data_original = extracted_data[
        extracted_data["found_keywords"].apply(len) > 0
    ].copy()
    logging.info(
        f"Keyword extraction completed. Keywords found in {filtered_data_original['found_keywords'].apply(len).sum()} cells."
    )

    # Ensure the local_id is maintained correctly
    filtered_data_original = filtered_data_original.reset_index(drop=True)

    #

    # Save the filtered dataset to a parquet file
    filtered_parquet_path = os.path.join(
        output_folder, f"{dataset_name.replace('/', '_')}_{split_name}_filtered.parquet"
    )
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filtered_parquet_path), exist_ok=True)

    filtered_data_original.to_parquet(filtered_parquet_path, index=False)
    logging.info(
        f"Filtered original data contains {len(filtered_data_original)} rows. Saved to {filtered_parquet_path}"
    )


def main(args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    mapper = DrugMapper(args.brand_to_generic_csv_path, args.generic_to_brand_csv_path)
    brand_to_generic_map = mapper.load_keywords("brand_to_generic")
    generic_to_brand_map = mapper.load_keywords("generic_to_brand")

    logging.info(f"Loaded brand to generic map: {brand_to_generic_map}")
    logging.info(f"Loaded generic to brand map: {generic_to_brand_map}")

    hf_datasets_questions = {
        "medmcqa": ["question"],
        "GBaker/MedQA-USMLE-4-options-hf": ["sent1"],
        # "hails/mmlu_no_train/clinical_knowledge": ["question"],
        # "hails/mmlu_no_train/professional_medicine": ["question"],
    }

    hf_datasets_choices = {
        "medmcqa": ["opa", "opb", "opc", "opd"],
        "GBaker/MedQA-USMLE-4-options-hf": ["ending0", "ending1", "ending3"],
        # "hails/mmlu_no_train/clinical_knowledge": ["choices"],
        # "hails/mmlu_no_train/professional_medicine": ["choices"],
    }

    if args.dataset_name.lower() == "questions":
        hf_datasets = hf_datasets_questions
    elif args.dataset_name.lower() == "choices":
        hf_datasets = hf_datasets_choices
    else:
        KeyError("Please select either 'questions' or 'choices' for dataset_name")

    datasets_to_process = hf_datasets.keys()

    for dataset_name in tqdm(datasets_to_process, desc="Overall Progress"):
        logging.info(f"Processing dataset: {dataset_name}")
        if dataset_name in hf_datasets:
            cols_of_interest = hf_datasets[dataset_name]
            logging.info(
                f"Processing dataset: {dataset_name} with columns: {cols_of_interest}"
            )

            dataset_splits = ["train", "dev", "validation", "test"]
            for split in dataset_splits:
                try:
                    if "hails/mmlu_no_train" in dataset_name:
                        subset = dataset_name.split("/")[-1]
                        dataset = load_dataset(
                            "hails/mmlu_no_train", subset, split=split
                        )
                    else:
                        dataset = load_dataset(dataset_name, split=split)

                    if not dataset:
                        logging.info(
                            f"No split named '{split}' found for dataset '{dataset_name}'"
                        )
                        continue

                    dataset_output_dir = os.path.join(
                        args.output_dir,
                        args.dataset_name,
                        dataset_name.replace("/", "_"),
                    )
                    os.makedirs(dataset_output_dir, exist_ok=True)

                    output_folder = os.path.join(dataset_output_dir, split)
                    os.makedirs(output_folder, exist_ok=True)

                    process_split_in_chunks(
                        dataset,
                        mapper,
                        brand_to_generic_map,
                        generic_to_brand_map,
                        cols_of_interest,
                        output_folder,
                        dataset_name,
                        split,
                        max_workers=args.max_workers,
                    )
                except ValueError as e:
                    logging.warning(
                        f"Could not load split '{split}' for dataset '{dataset_name}': {e}"
                    )
                except Exception as e:
                    logging.error(
                        f"An error occurred while processing split '{split}' for dataset '{dataset_name}': {e}"
                    )

        else:
            logging.warning(
                f"Dataset {dataset_name} is not recognized or not supported."
            )

    # Generate keyword counts CSV files
    brand_keywords = set(brand_to_generic_map.keys())
    generic_keywords = set(generic_to_brand_map.keys())
    generate_keyword_counts_csv(args.output_dir, brand_keywords, generic_keywords)


def count_keywords(merged_datasets, split, brand_keywords, generic_keywords):
    results = []
    for dataset_name, df in merged_datasets.items():
        if "found_keywords" in df.columns:
            # Flatten the list of keywords
            all_keywords = [
                keyword
                for sublist in df["found_keywords"].dropna()
                for keyword in sublist
            ]

            # Count total number of words (keywords)
            total_keyword_length = len(all_keywords)

            # Count brand and generic keywords
            brand_keyword_count = sum(
                1 for keyword in all_keywords if keyword in brand_keywords
            )
            generic_keyword_count = sum(
                1 for keyword in all_keywords if keyword in generic_keywords
            )

            # Collect the results
            results.append(
                {
                    "dataset": dataset_name,
                    "split": split,
                    "total_keywords": total_keyword_length,
                    "brand_keywords_count": brand_keyword_count,
                    "generic_keywords_count": generic_keyword_count,
                }
            )
        else:
            print(f"No 'found_keywords' column in {dataset_name}")

    return pd.DataFrame(results)


def generate_keyword_counts_csv(output_dir, brand_keywords, generic_keywords):
    results = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith("_filtered.parquet"):
                file_path = os.path.join(root, file)
                df = pd.read_parquet(file_path)
                dataset_name = os.path.basename(file_path).replace(
                    "_filtered.parquet", ""
                )
                split = os.path.basename(root)
                category = "choices" if "choices" in root else "questions"
                merged_datasets = {dataset_name: df}
                keywords_df = count_keywords(
                    merged_datasets, split, brand_keywords, generic_keywords
                )
                keywords_df["category"] = category  # Add the category to the dataframe
                results.append(keywords_df)

    final_df = pd.concat(results)
    final_csv_path = os.path.join(output_dir, "keywords_count.csv")
    final_df.to_csv(final_csv_path, index=False)
    logging.info(f"Keyword counts saved to {final_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process datasets with drug name mappings."
    )
    parser.add_argument(
        "--brand_to_generic_csv_path",
        type=str,
        default="RxNorm_eval/filtered_keywords.csv",
        help="Path to the CSV file containing brand to generic drug mappings.",
    )
    parser.add_argument(
        "--generic_to_brand_csv_path",
        type=str,
        default="RxNorm_eval/filtered_keywords.csv",
        help="Path to the CSV file containing generic to brand drug mappings.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pre_filter_datasets/col_counts/",
        help="Directory to save the processed datasets.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="choices",
        help="Specific dataset to process or 'all' for processing all datasets.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=30,
        help="Maximum number of worker processes to use for parallel processing.",
    )
    args = parser.parse_args()

    main(args)
