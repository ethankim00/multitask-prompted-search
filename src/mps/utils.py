import csv
import json
import logging
import os
import random
import requests
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from openmatch.arguments import DataArguments
from openmatch.dataset import InferenceDataset
from openmatch.utils import load_beir_positives
from src.mps.datasets import BEIR_DATASETS, CQA_DATASETS, OAG_DATASETS, OAGBeirConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BeirDatasetArguments:
    dataset: str = field(default=None, metadata={"help": "Beir Dataset to train on"})
    data_dir: str = "./data"


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


def download_dataset(dataset_args: BeirDatasetArguments) -> str:
    if dataset_args.dataset in BEIR_DATASETS:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
            dataset_args.dataset
        )
        out_dir = os.path.join(Path("./", dataset_args.data_dir))
        data_dir = util.download_and_unzip(url, out_dir)
    elif dataset_args.dataset in OAG_DATASETS:
        converter = OAGBeirConverter(
            data_dir=Path(dataset_args.data_dir).joinpath("oag_qa")
        )
        data_dir = converter.convert(dataset_args.dataset)
    elif dataset_args.dataset in CQA_DATASETS:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip"
        out_dir = os.path.join(Path("./", dataset_args.data_dir))
        data_dir = util.download_and_unzip(url, out_dir)
        data_dir = data_dir + "/" + dataset_args.dataset
    # elif dataset_args.dataset in TOP_LEVEL_OAG:
    #     s    pass
    # TODO logic for top level oag topics
    # Convert all lower level datasets
    # Compbine in a single directory, concatenative files and adjusting index mappings
    # 1 load
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


def construct_beir_dataset(dataset_name: str, tokenizer, split: str = "train"):
    data_dir = download_dataset(dataset_name)
    validate_data_splits(data_dir)
    qrels_dir = os.path.join(data_dir, "qrels")
    save_to = os.path.join(data_dir, "om_{}jsonl".format(split))
    qrels_file = os.path.join(qrels_dir, "{}.tsv".format(split))
    data_args = DataArguments(
        corpus_path=os.path.join(data_dir, "corpus.jsonl"),
        query_path=os.path.join(data_dir, "queries.jsonl"),
        query_template="<text>",
        doc_template="<title> [SEP] <text>",
    )

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
