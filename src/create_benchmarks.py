import os
import gc
import pandas as pd
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
    Sequence,
    load_dataset,
)  # Import load_dataset
from huggingface_hub import HfApi, HfFolder

# Retrieve token from environment variable
token = os.getenv("HF_TOKEN")

# Ensure the token is saved for authentication
if token:
    HfFolder.save_token(token)
else:
    raise ValueError("Hugging Face token not found in environment variables.")


def get_hf_features(dataset_name, split):
    """
    Get features from a Hugging Face dataset split.

    Parameters:
    dataset_name (str): The name of the dataset.
    split (str): The split to load the dataset from.

    Returns:
    dict: Features of the dataset split if successfully loaded, None otherwise.
    """
    print(f"Getting features for {dataset_name} split: {split}")
    if "hails" in dataset_name:
        local_dataset_name = dataset_name.split("/")
        try:
            dataset = load_dataset(
                "hails/mmlu_no_train", local_dataset_name[2], split=split
            )
            features = dataset.features
            print(f"Features for {dataset_name} ({split} split): {features}")
            # type of features
            print(f"Type of features: {type(features)}")

            return features
        except Exception as e:
            print(f"Could not load features for {dataset_name} ({split} split): {e}")
            return None
    try:
        dataset = load_dataset(dataset_name, split=split)
        features = dataset.features
        print(f"Features for {dataset_name} ({split} split): {features}")
        print(f"Type of features: {type(features)}")
        return features
    except Exception as e:
        print(f"Could not load features for {dataset_name} ({split} split): {e}")
        return None


def add_new_columns_to_features(features):
    """
    Add new columns to the existing features.

    Parameters:
    features (dict): Existing features of the dataset.

    Returns:
    dict: Updated features with new columns included.
    """
    print("Adding new columns to features...")
    print(f"Existing features: {features}")
    print(f"Type of features: {type(features)}")
    if isinstance(features, dict):
        features.update(
            {"found_keywords": Sequence(Value("string")), "local_id": Value("int64")}
        )
    print(f"Updated features: {features}")
    return features


def push_datasets_to_hub(
    base_path, namespace, dataset_name, hf_datasets, version, private=True
):
    """
    Push datasets to Hugging Face Hub.

    Parameters:
    base_path (str): The base directory where the datasets are stored.
    namespace (str): The Hugging Face namespace (username).
    dataset_name (str): The name of the dataset.
    hf_datasets (dict): A dictionary mapping dataset names to their features.
    version (str): Version identifier for the dataset.
    private (bool): Whether to make the dataset private on the Hugging Face Hub.
    """
    print(f"Pushing datasets to Hugging Face Hub for {dataset_name} version: {version}")
    # List of possible splits to infer features from
    possible_splits = ["train", "validation", "dev", "test"]

    # Try to infer features from each split
    features = None
    for split in possible_splits:
        features = get_hf_features(dataset_name, split)
        if features is not None:
            break

    if features is None:
        print(f"Could not infer features for any split of {dataset_name}")
        return

    # Add new columns to features
    features = add_new_columns_to_features(features)

    # Convert features to Features object
    features = Features(features)

    # Dictionary to hold datasets for each split
    dataset_splits = {}

    # Use inferred features across all local splits
    for split in possible_splits:
        # Adjust dataset name for local path
        local_dataset_name = dataset_name.replace("/", "_")

        if split == "test":
            version_suffix = f"{version}_filtered"
        else:
            version_suffix = version

        # Define dataset path
        split_path = os.path.join(
            base_path,
            local_dataset_name,
            split,
            f"{version_suffix}.parquet",
        )
        print(f"Loading {split} split from {split_path}...")

        if os.path.exists(split_path):
            print(f"Found {split} split at {split_path}")
            try:
                # Load local data
                local_df = pd.read_parquet(split_path)
                # print(f"First few rows of {split} split data:\n{local_df.head()}")
                # if empty df then skip and print error
                if local_df.empty:
                    print(f"Empty dataframe for {split} split")
                    print("\n" * 2)
                    continue
                local_data_as_dict = local_df.to_dict(orient="records")
                print(
                    f"Data as dictionary for {split} split:\n{local_data_as_dict[:5]}"
                )

                # Ensure features are printed correctly
                print(f"Features before creating Dataset: {features}")

                # Create a new Dataset object
                new_dataset = Dataset.from_pandas(
                    pd.DataFrame(local_data_as_dict), features=features
                )
                dataset_splits[split] = new_dataset

                print(f"Successfully loaded {split} split")

            except Exception as e:
                print(f"Error processing {split} split for {dataset_name}: {e}")
        else:
            print(f"{split} split file not found at {split_path}")

    if dataset_splits:
        new_dataset_dict = DatasetDict(dataset_splits)
        dataset_name_modified = (
            dataset_name.replace("/", "_").lower().replace("-", "_").replace(" ", "_")
        )
        repo_id = f"{namespace}/{dataset_name_modified}_{version}"

        print(f"Pushing {repo_id} to Hugging Face Hub...")
        try:
            new_dataset_dict.push_to_hub(repo_id, private=private, token=token)
            print(f"Successfully pushed {repo_id} to Hugging Face Hub")
            print("\n" * 10)
        except Exception as e:
            print(f"Error pushing {repo_id} to Hugging Face Hub: {e}")
            print("\n" * 10)


if __name__ == "__main__":
    # Define the base path and namespace
    base_path = "pre_filter_datasets"
    namespace = "AIM-Harvard"
    private = True

    hf_datasets = {
        "medmcqa": ["question", "opa", "opb", "opc", "opd"],
        "GBaker/MedQA-USMLE-4-options-hf": [
            "sent1",
            "ending0",
            "ending1",
            "ending2",
            "ending3",
        ],
    }

    dataset_names = list(hf_datasets.keys())
    for dataset_name in dataset_names:
        for version in [
            "generic_to_brand",
            "brand_to_generic",
            "original",
        ]:
            print(f"Processing dataset: {dataset_name} version: {version}")
            push_datasets_to_hub(
                base_path=base_path,
                namespace=namespace,
                dataset_name=dataset_name,
                hf_datasets=hf_datasets,
                version=version,
                private=private,
            )

        del hf_datasets[dataset_name]
        gc.collect()
