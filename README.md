# RABBITS: Robust Assessment of Biomedical Benchmarks Involving drug Term Substitutions

<!-- exclude_docs -->
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE.txt)
[![Arxiv](https://img.shields.io/badge/Arxiv-2406.12066-red)](https://arxiv.org/abs/2406.12066)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-RABBITS-green)](https://huggingface.co/spaces/AIM-Harvard/rabbits-leaderboard)


<!-- exclude_docs_end -->

**RABBITS** is a dataset transformation project that focuses on evaluating the robustness of language models in the medical **domain** through the substitution of brand names for generic drugs and vice versa. This evaluation is crucial as medical prescription errors, often related to incorrect drug naming, are a leading cause of morbidity and mortality.

![RABBITS Plot](rabbits_plot.png)

## Motivation

Medical knowledge is context-dependent and requires consistent reasoning across various natural language expressions of semantically equivalent phrases. This is particularly crucial for drug names, where patients often use brand names like Advil or Tylenol instead of their generic equivalents. To study this, we create a new robustness dataset, RABBITS, to evaluate performance differences on medical benchmarks after swapping brand and generic drug names using physician expert annotations.

By modifying common medical benchmarks such as MedMCQA and MedQA, and replacing drug names, we aim to:

- Test models' robustness in clinical knowledge comprehension.
- Detect signs of dataset contamination or model bias.
- Highlight the importance of precise drug nomenclature in medical AI applications.

Our findings reveal a consistent performance drop ranging from 1-10% when drug names are swapped, suggesting potential issues in the model's ability to generalize and highlighting the need for robustness in medical AI applications.

## Setup

### Environment

The codebase is designed to run with Python 3.9. Here are the steps to set up the environment:

```bash
conda create -n rabbits python=3.9
conda activate rabbits
pip install -r requirements.txt
```

## Datasets

The following datasets are transformed and analyzed in the RABBITS project:

1. MedQA (medqa)
2. MedMCQA (medmcqa)

Each dataset is processed to create three subsets: a subset that is filtered to questions that contain a drug keyword then two other variants, one with brand names replaced by generic names and another with generic names replaced by brand names. These transformations test the model's ability to maintain accuracy despite changes in clinically equivalent terminology.


## Drug Names
`data/generic_to_brand.csv` - contains the list of generic to brand drug names searched for in the dataset
`src/get_drug_list.py` - creates a list of brand and generic drug pairs using the `RxNorm` dataset (should be in data/RxNorm). 

## Keyword Extraction
`src/drug_mapping` - contains the code to create brand-generic drug mappings
`src/run_filtering.py` - runs the keyword extraction and filters the datasets to only entries that contain either a brand or generic drug name
`src/run_replacement.py` - runs the keyword replacement on the filtered datasets
`src/counts_per_col.py` - counts the number of times each word appears in each column of the dataset e.g. question vs answers

## Annotations
`src/get_rag_names.py` - Creates api calls to cohere to check brand names for each generic drug which was used during the quality assurance process.
`data/annotated_medqa_new.csv` and `data/annotated_medmcqa_new.csv` - The annotated questions for correct replacement of drug names that is used to filter the dataset for creating the final dataset. 
`src/generate_final_questions.py` - filters the drug names to only include those questions that were accepted during the quality assurance process

Each row in the annotation files represents a question and includes the following key information:

- Original question and answer choices
- Generic-to-brand (g2b) transformed question and answer choices
- Found keywords in the original question
- Decision to keep or drop the question
- Any additional comments

### Example Annotation

Here's an example of how the annotations are structured:

```
id,question_orig,opa_orig,opb_orig,opc_orig,opd_orig,cop_orig,choice_type_orig,exp_orig,subject_name_orig,topic_name_orig,found_keywords_orig,local_id_orig,Unnamed: 13,question_g2b,opa_g2b,opb_g2b,opc_g2b,opd_g2b,cop_g2b,choice_type_g2b,exp_g2b,subject_name_g2b,topic_name_g2b,found_keywords_g2b,local_id_g2b,Unnamed: 26,keep/drop,comments
006acfff-dc8f-4bb5-97b2-e26144c56483,PGE1 analogue is ?,Carboprost,Alprostadil,Epoprostenol,Dinoprostone,-1,single,NaN,Pharmacology,NaN,['carboprost' 'dinoprostone' 'alprostadil' 'epoprostenol'],4101,NaN,PGE1 analogue is ?,hemabate,caverject,flolan,cervidil,-1,single,NaN,Pharmacology,NaN,['carboprost' 'dinoprostone' 'alprostadil' 'epoprostenol'],4101,NaN,keep,NaN
```

In this example:
- The original question asks about PGE1 analogues, with generic drug names as options.
- The g2b transformation replaces these with corresponding brand names (e.g., Carboprost → hemabate).
- The found keywords are listed, showing which drug names were identified.
- The 'keep/drop' column indicates that this question should be kept in the final dataset.


## Results
`src/create_benchmarks.py` - creates a benchmark for Hugging Face and pushes it to the hub
`src/create_plots.py` - contains the code to create the figures in the paper



# Citing
```
@inproceedings{
  anonymous2024language,
  title={Language Models are Surprisingly Fragile to Drug Names in Biomedical Benchmarks},
  author={Anonymous},
  booktitle={The 2024 Conference on Empirical Methods in Natural Language Processing},
  year={2024},
  url={https://openreview.net/forum?id=T5taTxsNb3}
}
```
