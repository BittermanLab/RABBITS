import pandas as pd
import random
from typing import Dict, Any, List
import re
import json
import os
from tqdm.auto import tqdm

import logging

logger = logging.basicConfig(level=logging.INFO)


class DrugMapper:
    def __init__(
        self, brand_to_generic_csv: str, generic_to_brand_csv: str, seed: int = 42
    ):
        """
        Initializes the DrugMapper class with paths to the CSV files and a seed for random operations.
        """
        self.seed = seed
        self.brand_to_generic_df = pd.read_csv(brand_to_generic_csv)
        self.generic_to_brand_df = pd.read_csv(generic_to_brand_csv)

    def load_keywords(self, mapping_type: str) -> Dict[str, str]:
        """
        Load keywords mapping from the CSV files based on the mapping type.
        """
        if mapping_type == "brand_to_generic":
            return self._load_map(self.brand_to_generic_df, "brand", "generic")
        elif mapping_type == "generic_to_brand":
            return self._load_map(self.generic_to_brand_df, "generic", "brand")
        else:
            raise ValueError(
                "Invalid mapping type. Use 'brand_to_generic' or 'generic_to_brand'."
            )

    def _load_map(
        self, df: pd.DataFrame, key_col: str, value_col: str
    ) -> Dict[str, str]:
        """
        Create a mapping dictionary from the dataframe.
        """
        df = df.dropna(
            subset=[key_col, value_col]
        )  # Drop rows where either key_col or value_col is NaN
        mapping = {}
        for key, value in zip(df[key_col], df[value_col]):
            key = key.strip().lower()  # Normalize keys to lowercase and strip spaces
            value = (
                value.strip().lower()
            )  # Normalize values to lowercase and strip spaces
            if key and value:
                if key not in mapping:
                    mapping[key] = value
                else:
                    logging.warning(
                        f"Duplicate key '{key}' found in mapping. Existing value: '{mapping[key]}', new value: '{value}'"
                    )
        return mapping

    def load_all_keywords_list(self) -> List[str]:
        """
        Load and deduplicate keywords from both brand to generic and generic to brand mappings.
        """
        btog = self.brand_to_generic_df["brand"].tolist()
        gtob = self.generic_to_brand_df["generic"].tolist()

        # Deduplicate keywords
        keywords = list(set(map(str.lower, btog + gtob)))
        return keywords
