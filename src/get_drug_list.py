import pandas as pd


def load_rxnconso(file_path):
    """
    Load the RXNCONSO.RRF file into a pandas DataFrame.

    Args:
    file_path (str): The path to the RXNCONSO.RRF file.

    Returns:
    pd.DataFrame: A DataFrame containing the RXNCONSO data.
    """
    column_names = [
        "RXCUI",
        "LAT",
        "TS",
        "LUI",
        "STT",
        "SUI",
        "ISPREF",
        "RXAUI",
        "SAUI",
        "SCUI",
        "SDUI",
        "SAB",
        "TTY",
        "CODE",
        "STR",
        "SRL",
        "SUPPRESS",
        "CVF",
    ]
    df = pd.read_csv(file_path, sep="|", names=column_names, index_col=False)
    return df


def load_rxnrel(file_path):
    """
    Load the RXNREL.RRF file into a pandas DataFrame.

    Args:
    file_path (str): The path to the RXNREL.RRF file.

    Returns:
    pd.DataFrame: A DataFrame containing the RXNREL data.
    """
    column_names = [
        "RXCUI1",
        "RXAUI1",
        "STYPE1",
        "REL",
        "RXCUI2",
        "RXAUI2",
        "STYPE2",
        "RELA",
        "RUI",
        "SRUI",
        "SAB",
        "SL",
        "DIR",
        "RG",
        "SUPPRESS",
        "CVF",
    ]
    df = pd.read_csv(file_path, sep="|", names=column_names, index_col=False)
    return df


def is_excluded(name, overlapping_words):
    """
    Check if a drug name should be excluded based on certain criteria.

    Args:
    name (str): The drug name to check.
    overlapping_words (list): A list of words that should cause exclusion if present in the name.

    Returns:
    bool: True if the name should be excluded, False otherwise.
    """
    if len(name.split()) > 1:
        return True
    if any(char for char in name if not char.isalnum() and char not in {" ", "-"}):
        return True
    if any(char.isdigit() for char in name):
        return True
    if any(
        word in name.lower()
        for word in ["obsolete", "withdrawn", "discontinued", "containing"]
    ):
        return True
    if any(word in name.lower() for word in overlapping_words):
        return True
    return False


if __name__ == "__main__":
    # Load the data
    rxnconso_df = load_rxnconso("../data/RxNorm/RXNCONSO.RRF")
    rxnrel_df = load_rxnrel("../data/RxNorm/RXNREL.RRF")

    # Filter rxnconso_df for IN and BN entries
    rxnconso_filtered = rxnconso_df[rxnconso_df["TTY"].isin(["IN", "PT", "BN"])]

    # Join rxnrel_df with rxnconso_filtered
    rxnrel_df = rxnrel_df.merge(
        rxnconso_filtered[["RXCUI", "TTY", "STR"]],
        left_on="RXCUI1",
        right_on="RXCUI",
        how="left",
        suffixes=("", "_tmp1"),
    )
    rxnrel_df.rename(columns={"TTY": "TTY1", "STR": "STR1"}, inplace=True)

    rxnrel_df = rxnrel_df.merge(
        rxnconso_filtered[["RXCUI", "TTY", "STR"]],
        left_on="RXCUI2",
        right_on="RXCUI",
        how="left",
        suffixes=("", "_tmp2"),
    )
    rxnrel_df.rename(columns={"TTY": "TTY2", "STR": "STR2"}, inplace=True)

    # Handle missing RXCUI by joining on RXAUI
    rxnrel_df.loc[rxnrel_df["TTY1"].isnull(), ["TTY1", "STR1"]] = (
        rxnrel_df.loc[rxnrel_df["TTY1"].isnull()]
        .merge(
            rxnconso_filtered[["RXAUI", "TTY", "STR"]],
            left_on="RXAUI1",
            right_on="RXAUI",
            how="left",
        )[["TTY", "STR"]]
        .values
    )

    rxnrel_df.loc[rxnrel_df["TTY2"].isnull(), ["TTY2", "STR2"]] = (
        rxnrel_df.loc[rxnrel_df["TTY2"].isnull()]
        .merge(
            rxnconso_filtered[["RXAUI", "TTY", "STR"]],
            left_on="RXAUI2",
            right_on="RXAUI",
            how="left",
        )[["TTY", "STR"]]
        .values
    )

    # Filter for relation of tradename_of
    rxnrel_df = rxnrel_df[rxnrel_df["RELA"].isin(["tradename_of"])]

    # Get pairs
    tradename_pairs = rxnrel_df[rxnrel_df["RELA"] == "tradename_of"][["STR1", "STR2"]]
    tradename_pairs.rename(
        columns={"STR1": "Ingredient", "STR2": "Tradename"}, inplace=True
    )

    # Normalize strings and remove duplicates
    tradename_pairs["Ingredient_norm"] = (
        tradename_pairs["Ingredient"].str.lower().str.strip()
    )
    tradename_pairs["Tradename_norm"] = (
        tradename_pairs["Tradename"].str.lower().str.strip()
    )

    tradename_pairs_deduped = tradename_pairs.drop_duplicates(
        subset=["Ingredient_norm", "Tradename_norm"]
    )
    tradename_pairs_deduped = tradename_pairs_deduped[
        ["Ingredient_norm", "Tradename_norm"]
    ].reset_index(drop=True)
    tradename_pairs_deduped.rename(
        columns={"Ingredient_norm": "generic", "Tradename_norm": "brand"}, inplace=True
    )

    # Remove rows with empty generic or brand names
    tradename_pairs_deduped["generic"].fillna("", inplace=True)
    tradename_pairs_deduped["brand"].fillna("", inplace=True)
    tradename_pairs_deduped = tradename_pairs_deduped[
        (tradename_pairs_deduped["generic"] != "")
        & (tradename_pairs_deduped["brand"] != "")
    ].reset_index(drop=True)

    # Define exclusion criteria
    overlapping_words = [
        "today",
        "thrive",
        "program",
        "react",
        "perform",
        "tomorrow",
        "bronchial",
        "copd",
        "duration",
        "matrix",
        "blockade",
        "sustain",
        "overtime",
        "android",
        "suppressor",
        "nephron",
        "alcohol",
        "liver",
        "thyroid",
        "potassium",
        "prothrombin",
        "alanine",
        "water",
        "oxygen",
        "peanut",
        "urea",
        "nitrogen",
        "acetylcholine",
        "lactate",
        "glucose",
        "arginine",
        "glutamine",
        "testosterone",
        "tyrosine",
        "ethanol",
        "progesterone",
        "isoleucine",
        "choline",
        "glycine",
        "glutamate",
        "amylase",
        "leucine",
        "phenylalanine",
        "starch",
        "sulfur",
        "phosphorus",
        "cysteine",
        "sucrose",
    ]

    # Apply exclusion criteria
    filtered_tradename_pairs = tradename_pairs_deduped[
        ~tradename_pairs_deduped["generic"].apply(
            is_excluded, args=(overlapping_words,)
        )
        & ~tradename_pairs_deduped["brand"].apply(
            is_excluded, args=(overlapping_words,)
        )
    ]

    # Create generic to brand mapping
    generic_to_brand = (
        filtered_tradename_pairs.groupby("generic")["brand"].apply(list).reset_index()
    )
    generic_to_brand["brand"] = [i[0] for i in generic_to_brand["brand"]]

    # Save results
    generic_to_brand.to_csv("generic_to_brand_mapping.csv", index=False)
    filtered_tradename_pairs.to_csv("filtered_tradename_pairs.csv", index=False)

    print(
        "Processing complete. Results saved to 'generic_to_brand_mapping.csv' and 'filtered_tradename_pairs.csv'."
    )
