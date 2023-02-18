# Adapted from Tevatron (https://github.com/texttron/tevatron)

import csv
import json
import os
import random
from datetime import datetime
from functools import partial

from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from openmatch.arguments import DataArguments
from openmatch.dataset import InferenceDataset
from openmatch.utils import load_beir_positives, load_positives


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


def construct_beir_training_datast(dataset_name: str, tokenizer):
    data_dir = load_dataset(dataset_name)
    validate_data_splits(data_dir)
    qrels_dir = os.path.join(data_dir, "qrels")
    save_to = os.path.join(data_dir, "om_train.jsonl")
    qrels_file = os.path.join(qrels_dir, "train.tsv")
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
    return save_to
