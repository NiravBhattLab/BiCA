# BiCA: Effective Biomedical Dense Retrieval with Citation-Aware Hard Negatives

This repository contains the code for a pipeline to train and evaluate a biomedical retrieval model using the GPL framework. The pipeline consists of three main stages: building a corpus with hard negatives, training the model, and evaluating its performance on various benchmark datasets. All fine-tuned models and created datasets are available in this [HuggingFace Collection](https://huggingface.co/collections/bisectgroup/bica-65f65bccb76ab963c3cc8f92).

## Prerequisites

Before running any of the scripts, ensure you have the necessary libraries installed. You can install them using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Step 1: Build Corpus

This stage involves generating a dataset of queries, positive passages, and hard negatives from the PubMed abstract dataset. This is accomplished in two steps.

### 1.1. Generate 2-Hop Citation Graphs

The `build-corpus/pubmed-parser.py` script downloads PubMed abstracts and constructs 2-hop citation graphs. For each starting abstract, it fetches the abstracts of its cited papers (1-hop) and the papers they cite (2-hop).

To run this script:

```bash
python build-corpus/pubmed-parser.py
```

This will create the following file:

*   `2hop-citation-graphs.jsonl`: Contains the 2-hop citation graphs, with each line representing a starting PMID and its corresponding 1-hop and 2-hop abstracts.

### 1.2. Generate Hard Negatives

The `build-corpus/pubmed-query-scoring.py` script takes the citation graphs from the previous step and generates queries and hard negatives. It uses the [T5 Doc2Query model](https://huggingface.co/doc2query/all-t5-base-v1) to create a query for each positive abstract and then traverses the citation graph to find diverse hard negatives.

To run this script:

```bash
python build-corpus/pubmed-query-scoring.py
```

This will produce the following file, which will be used for training:

*   `hard-negatives-traversal.jsonl`: A JSONL file where each line contains a query, a positive passage, and a list of hard negative passages.

## Step 2: Fine-tune the Model

The `train.py` script fine-tunes the [gte-models](https://huggingface.co/thenlper/gte-base) using the data generated in the previous step. It uses a multiple negatives ranking loss to train the model to distinguish between positive and negative passages for a given query.

To start the training process:

```bash
python train.py
```

The script will save the fine-tuned model to the following directory:

*   `output/`: This folder will contain the trained model artifacts. The specific sub-folder will depend on the `MODEL_NAME` set in the `train.py` script. For example, if `MODEL_NAME` is `'thenlper/gte-base'`, the model will be saved in `output/gte/`.

## Step 3: Evaluate the Model

The final stage is to evaluate the performance of the trained model on various benchmark datasets. The evaluation scripts use the `beir` library, and the datasets are available from the [BEIR GitHub repository](https://github.com/beir-cellar/beir). Make sure to download the necessary datasets and place them in the `eval_datasets/` directory. For LoTTE the datasets are downloaded from [IR Datasets](https://ir-datasets.com/lotte.html).

### 3.1. Evaluation on BEIR


To run the evaluation:

```bash
python eval/beir-evaluation.py
```

### 3.2. CQADupStack Evaluation

The `eval/cqadupstack.py` script evaluates the model on the CQADupStack benchmark, which consists of sub-datasets from different domains.

To run this evaluation:

```bash
python eval/cqadupstack.py
```

### 3.3. LoTTE Evaluation

The `eval/lotte-evaluation.py` script evaluates the model on the LoTTE benchmark.

To run the LoTTE evaluation, you need to provide the path to the model and the data directories:

```bash
python eval/lotte-evaluation.py --model_path output/gte --data_dir eval_datasets/lotte --rankings_dir rankings --split test
```
