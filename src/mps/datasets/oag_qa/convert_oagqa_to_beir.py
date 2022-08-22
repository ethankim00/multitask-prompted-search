import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import *

import json
import logging

logger = logging.getLogger(name=__name__)


def save_jsonl(output_dir: str = "./data", filename: str = "corpus.jsonl", data: List[Dict] = [])-> None:
    output_dir = Path(output_dir).joinpath(filename)
    if output_dir.is_file():
        raise FileExistsError("File alredy exists")
    with open(output_dir, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")
    return output_dir

def mine_positive_document_ids(answers: List[str], oag_corpus_df: pd.DataFrame):
    positive_ids = []
    for answer in answers:
        try:
            doc_ids =list(oag_corpus_df.loc[oag_corpus_df[2] == answer][0].values)
        except Exception as e:
            print(e)
            doc_ids = []
        positive_ids += doc_ids
    return positive_ids

def _extract_qrels(query_df: pd.DataFrame)-> pd.DataFrame:
    query_ids = []
    corpus_ids = []
    for i, row in query_df.iterrows():
        corpus_ids += row["positive_ids"]
        query_ids += [i for _ in row["positive_ids"]]
        
    qrels = pd.DataFrame()
    qrels["query-id"] = query_ids
    qrels["corpus-id"] = corpus_ids
    qrels["score"] = [1 for _ in range(len(qrels))]
    return qrels

def extract_qrels(oag_query_df: pd.DataFrame, oag_corpus_df: pd.DataFrame) -> pd.DataFrame:
    oag_query_df["positive_ids"]  = oag_query_df[1].apply(lambda x:  mine_positive_document_ids(eval(x), oag_corpus_df))
    qrels = _extract_qrels(oag_query_df)
    return qrels

def extract_queries(oag_query_df: pd.DataFrame) -> List[Dict]:
    return [{"_id": str(idx) , "text" : text, "metadata": {}} for idx, text in zip(oag_query_df.index, oag_query_df[0])]
    

def train_test_val_split(df: pd.DataFrame, train_split: float =  0.8, dev_split: float = 0.1):
    train, validate, test =np.split(df.sample(frac=1, random_state=42),  [int(train_split*len(df)), int((train_split + dev_split)*len(df))])
    return train, validate, test


class OAGBeirConverter:
    
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir
        

    def _load_oag_data_raw(self, topic_name: str, file_type: str, data_dir: Optional[str] = None)-> Union[pd.DataFrame, List]:
        data_dir = data_dir if data_dir else self.data_dir
        file_type_mappings = {
            "corpus" : "-papers-10k.tsv",
            "queries" : "-questions.tsv",
            "qrels": "-train.json",
        }
        path = Path(data_dir).joinpath(topic_name + file_type_mappings[file_type])
        if file_type in ["corpus", "queries"]:
            data = pd.read_csv(path, header = None, sep = "\t")
        else:
            with open(path, "r") as f:
                data = json.load(f)
        return data
    

    def _load_all_files(self, topic_name: str) -> Dict:
        result = {file_type: self._load_oag_data_raw(topic_name, file_type) for file_type in ["corpus", "queries", "qrels"]}
        return result
    
    
    @staticmethod
    def convert_oag_beir_corpus(oag_corpus: pd.DataFrame)-> List[Dict]: 
        converted_corpus = [{"_id":  str(_id), "title": "", "text" : str(title) + " " + str(text)} for _id, text, title in zip(oag_corpus[0], oag_corpus[1], oag_corpus[2])]
        return converted_corpus
        
        
        
    def convert(self, dataset: str, output_dir: str = "./data/",  data_dir: Optional[str] = None) -> str:
        data_dir = data_dir if data_dir else self.data_dir
        dataset_folder = Path(output_dir).joinpath("beir_" + dataset)
        if dataset_folder.joinpath("corpus.jsonl").is_file():
            logger.info("Dataset already converted")
            return dataset_folder
        oag_data = self._load_all_files(topic_name = dataset)
        beir_corpus = self.convert_oag_beir_corpus(oag_data["corpus"])
        beir_queries = extract_queries(oag_data["queries"])
        beir_qrels = extract_qrels(oag_data["queries"], oag_data["corpus"])
        train, validate, test = train_test_val_split(beir_qrels)
        dataset_folder.mkdir(parents=True, exist_ok=True)
        qrels_folder = dataset_folder.joinpath("qrels")
        qrels_folder.mkdir(parents=True, exist_ok=True)
        save_jsonl(dataset_folder, "corpus.jsonl", beir_corpus)
        save_jsonl(dataset_folder, "queries.jsonl", beir_queries)
        train.to_csv(qrels_folder.joinpath("train.tsv"), sep="\t", index = False)
        validate.to_csv(qrels_folder.joinpath("dev.tsv"), sep="\t", index = False)
        test.to_csv(qrels_folder.joinpath("test.tsv"), sep="\t", index = False)                          
        return dataset_folder