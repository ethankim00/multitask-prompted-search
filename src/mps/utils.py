import csv
import json
import logging
import os
import random
import requests
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path

from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from openmatch.arguments import DataArguments
from openmatch.dataset import InferenceDataset
from openmatch.utils import load_beir_positives
from src.mps.datasets import (
    BEIR_DATASETS,
    CQA_DATASETS,
    OAG_DATASETS,
    TOP_LEVEL_OAG_DATASETS,
    OAGBeirConverter,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BEIRDataArguments(DataArguments):
    train_dataset: str = field(
        default=None, metadata={"help": "name of train dataset to use"}
    )
    eval_dataset: str = field(
        default=None, metadata={"help": "name of eval dataset to use"}
    )


def download_url(url: str, save_path: str, chunk_size: int = 1024):
    """Download url with progress bar using tqdm
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    Args:
        url (str): downloadable url
        save_path (str): local path to save the downloaded file
        chunk_size (int, optional): chunking of files. Defaults to 1024.
    """
    r = requests.get(url, stream=True)
    total = int(r.headers.get("Content-Length", 0))
    with open(save_path, "wb") as fd, tqdm(
        desc=save_path,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=chunk_size,
    ) as bar:
        for data in r.iter_content(chunk_size=chunk_size):
            size = fd.write(data)
            bar.update(size)


def unzip(zip_file: str, out_dir: str):
    zip_ = zipfile.ZipFile(zip_file, "r")
    zip_.extractall(path=out_dir)
    zip_.close()


def download_and_unzip(url: str, out_dir: str, chunk_size: int = 1024) -> str:

    os.makedirs(out_dir, exist_ok=True)
    dataset = url.split("/")[-1]
    zip_file = os.path.join(out_dir, dataset)

    if not os.path.isfile(zip_file):
        logger.info("Downloading {} ...".format(dataset))
        download_url(url, zip_file, chunk_size)

    if not os.path.isdir(zip_file.replace(".zip", "")):
        logger.info("Unzipping {} ...".format(dataset))
        unzip(zip_file, out_dir)

    return os.path.join(out_dir, dataset.replace(".zip", ""))


def group_top_level_oag_topics(dataset: str) -> str:
    """
    Generate a BEIR dataset combining all lower level topics of a top level OAG topic.

    Args:
        dataset (str): Name of OAG top level dataset

    Returns:
        str: Path to new combined dataset forlder in BEIR format
    """
    lower_level_topics = TOP_LEVEL_OAG_DATASETS[dataset]

    def load_corpus_and_queries(lower_level_datasets: List[str]) -> List[Dict]:
        corpus = []
        queries = []
        _data_dir = "./data"
        for dataset in lower_level_datasets:
            converter = OAGBeirConverter(data_dir=Path(_data_dir).joinpath("oag_qa"))
            data_dir = converter.convert(
                dataset,
            )
            with open(f"{data_dir}/corpus.jsonl") as f:
                corpus.extend(
                    [
                        {**doc, "_id": f"{dataset}_{doc['_id']}"}
                        for doc in [json.loads(l) for l in f]
                    ]
                )
            with open(f"{data_dir}/queries.jsonl") as f:
                queries.extend(
                    [
                        {**query, "_id": f"{dataset}_{query['_id']}"}
                        for query in [json.loads(l) for l in f]
                    ]
                )
        return corpus, queries

    corpus, queries = load_corpus_and_queries(lower_level_topics)

    def load_qrels(
        lower_level_datasets: List[str], split: str = "train"
    ) -> pd.DataFrame:
        qrels = []
        for dataset in lower_level_datasets:
            data_dir = Path("./data").joinpath("beir_{}".format(dataset))
            df = pd.read_csv(f"{data_dir}/qrels/{split}.tsv", sep="\t")
            df["query-id"] = df["query-id"].apply(lambda x: f"{dataset}_{x}")
            df["corpus-id"] = df["corpus-id"].apply(lambda x: f"{dataset}_{x}")
            qrels.append(df)
        qrels = pd.concat(qrels)
        return qrels

    train_qrels = load_qrels(lower_level_topics, "train")
    dev_qrels = load_qrels(lower_level_topics, "dev")
    test_qrels = load_qrels(lower_level_topics, "test")
    output_dir = f"./data/oag_qa_top_level/{dataset}"
    os.makedirs(output_dir, exist_ok=True)
    qrels_dir = Path(output_dir).joinpath("qrels")
    os.makedirs(qrels_dir, exist_ok=True)
    train_qrels.to_csv(f"{output_dir}/qrels/train.tsv", sep="\t", index=False)
    dev_qrels.to_csv(f"{output_dir}/qrels/dev.tsv", sep="\t", index=False)
    test_qrels.to_csv(f"{output_dir}/qrels/test.tsv", sep="\t", index=False)
    # WRite query and corpus entries to jsonl format
    with open(f"{output_dir}/corpus.jsonl", "w") as f:
        for doc in corpus:
            json.dump(doc, f)
            f.write("\n")
    with open(f"{output_dir}/queries.jsonl", "w") as f:
        for query in queries:
            json.dump(query, f)
            f.write("\n")
    return output_dir


def download_dataset(dataset: str) -> str:
    data_dir = "./data"
    if dataset in BEIR_DATASETS:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
            dataset
        )
        out_dir = os.path.join(Path("./", data_dir))
        data_dir = download_and_unzip(url, out_dir)
    elif dataset in OAG_DATASETS:
        converter = OAGBeirConverter(data_dir=Path(data_dir).joinpath("oag_qa"))
        data_dir = converter.convert(dataset)
    elif dataset in CQA_DATASETS:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip"
        out_dir = os.path.join(Path("./", data_dir))
        data_dir = download_and_unzip(url, out_dir)
        data_dir = data_dir + "/" + dataset
    elif dataset in TOP_LEVEL_OAG_DATASETS:
        data_dir = group_top_level_oag_topics(dataset)
    return data_dir


def validate_data_splits(data_path: str):
    """
    Checks if the data splits are already present in the data_path. If not, it creates them.

    Args:
        data_path (str): Path to the data directory
    """
    data_path = Path(data_path)
    base_path = data_path.joinpath("qrels")
    train_path = base_path.joinpath("train.tsv")
    dev_path = base_path.joinpath("dev.tsv")
    test_path = base_path.joinpath("test.tsv")

    paths = [train_path, dev_path, test_path]

    if all([path.is_file() for path in paths]):
        train_df = pd.read_csv(train_path, delimiter="\t")
        if len(train_df) > 10000:  # limit to 10000 training examples
            logger.info("Limiting Train Dataset")
            train_df = train_df[0:10000]
            train_df.to_csv(train_path, sep="\t", index=False)
        return

    if test_path.is_file():
        if not any([train_path.is_file(), dev_path.is_file()]):
            logger.info("Creating training and validation splits from test dataset")
            df = pd.read_csv(test_path, delimiter="\t")
            train, validate, test = np.split(
                df.sample(frac=1, random_state=42),
                [int(0.8 * len(df)), int(0.9 * len(df))],
            )
            train.to_csv(train_path, sep="\t", index=False)
            validate.to_csv(dev_path, sep="\t", index=False)
            test.to_csv(test_path, sep="\t", index=False)
            return

        if train_path.is_file() and not dev_path.is_file():
            logger.info("Creating validation split from train dataset")
            df = pd.read_csv(
                data_path.joinpath("qrels").joinpath("train.tsv"), delimiter="\t"
            )
            train, validate = np.split(
                df.sample(frac=1, random_state=42), [int(0.8 * len(df))]
            )
            train.to_csv(train_path, sep="\t", index=False)
            validate.to_csv(dev_path, sep="\t", index=False)
            return

        if not train_path.is_file() and dev_path.is_file():
            logger.info("Creating training split from test and dev datasets")
            test_df = pd.read_csv(
                data_path.joinpath("qrels").joinpath("test.tsv"), delimiter="\t"
            )
            val_df = pd.read_csv(
                data_path.joinpath("qrels").joinpath("dev.tsv"), delimiter="\t"
            )
            df = pd.concat([test_df, val_df])
            train, validate, test = np.split(
                df.sample(frac=1, random_state=42),
                [int(0.8 * len(df)), int(0.9 * len(df))],
            )
            train.to_csv(train_path, sep="\t", index=False)
            validate.to_csv(dev_path, sep="\t", index=False)
            test.to_csv(test_path, sep="\t", index=False)
            return


def get_positive_and_negative_samples(query_dataset, corpus_dataset, qrel, qid):
    origin_positives = qrel.get(qid, [])
    if (len(origin_positives)) >= 1:
        item = process_one(query_dataset, corpus_dataset, qid, origin_positives)
        if item["positives"]:
            return item
        else:
            return None


def process_one(query_dataset, corpus_dataset, q, poss):
    try:
        train_example = {
            "query": query_dataset[q]["input_ids"],
            "positives": [
                corpus_dataset[p]["input_ids"]
                for p in poss
                if corpus_dataset[p]["input_ids"]
            ],
            "negatives": [],
        }
    except:
        return {"positives": False}
    return train_example


def filter_metadata(query_path):
    """
    Load the jsonl file and remove the metadata column from the dictionary.

    Args:
        query_path (Path): Path to the jsonl file
    """
    with open(query_path, "r") as f:
        data = [json.loads(line) for line in f]
    for item in data:
        item.pop("metadata", None)
    # delete the old jsonl file
    os.remove(query_path)
    with open(query_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def construct_beir_dataset(dataset_name: str, tokenizer, split: str = "train"):
    data_dir = download_dataset(dataset_name)
    validate_data_splits(data_dir)
    qrels_dir = os.path.join(data_dir, "qrels")
    save_to = os.path.join(data_dir, "om_{}.jsonl".format(split))
    qrels_file = os.path.join(qrels_dir, "{}.tsv".format(split))
    data_args = DataArguments(
        corpus_path=os.path.join(data_dir, "corpus.jsonl"),
        query_path=os.path.join(data_dir, "queries.jsonl"),
        query_template="<text>",
        doc_template="<title> [SEP] <text>",
    )
    if dataset_name in ["fever", "hotpotqa"]:
        # filter the metadata column from the queries jsonl file
        filter_metadata(data_args.query_path)

    query_dataset = InferenceDataset.load(
        tokenizer=tokenizer,
        data_args=data_args,
        is_query=True,
        full_tokenization=False,
        stream=False,
    )
    corpus_dataset = InferenceDataset.load(
        tokenizer=tokenizer,
        data_args=data_args,
        is_query=False,
        full_tokenization=False,
        stream=False,
    )
    qrel = load_beir_positives(qrels_file)

    get_positive_samples_partial = partial(
        get_positive_and_negative_samples,
        query_dataset,
        corpus_dataset,
        qrel,
    )

    save_dir = os.path.split(save_to)[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    query_list = []
    for query_id, doc_id in qrel.items():
        query_list.append(query_id)
    contents = list(
        tqdm(map(get_positive_samples_partial, query_list), total=len(query_list))
    )

    with open(save_to, "w") as f:
        for result in tqdm(contents):
            if result is not None:
                f.write(json.dumps(result))
                f.write("\n")
    return save_dir
