import pandas as pd
import json
from tqdm import tqdm
import cohere

# Set pandas display options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

if __name__ == "__main__":
    # read in generic_to_brand
    generic_to_brand = pd.read_csv("../data/generic_to_brand.csv")

    # Use Cohere API to verify and improve brand names
    co = cohere.Client(api_key="your_api_key_here")
    results = []

    for brand, gen in tqdm(generic_to_brand[["brand", "generic"]].values):
        message = f"generic: {gen} brand: {brand} \n please return a python list the best brand drugs that are not Combination drugs mainly contains the generic, make sure they are used for human."
        response = co.chat(
            model="command-r-plus",
            message=message,
            temperature=0.3,
        )
        results.append(
            {
                "brand": brand,
                "generic": gen,
                "message": message,
                "response": response.text,
            }
        )

    # Save results
    with open("../data/rag.json", "w") as f:
        json.dump(results, f)

    rag_df = pd.DataFrame(results)
    rag_df.to_csv("../data/rag.csv", index=False)

    print("Processing complete. Results saved to 'data/rag.json' and 'data/rag.csv'.")
