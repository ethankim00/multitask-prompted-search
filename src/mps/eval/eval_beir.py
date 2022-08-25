
import argparse
import logging
import pathlib, os
import random
from transformers import HfArgumentParser
from typing import Dict, List

from sentence_transformers.evaluation import InformationRetrievalEvaluator

from src.mps.utils import  download_dataset, BeirDatasetArguments
from src.mps.models import load_model

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os

from dataclasses import dataclass, field, asdict

from pathlib import Path
logger = logging.getLogger(__name__)

wandb_logging = False



@dataclass
class EvaluationArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    dataset: str = field(metadata={"help": "Beir Dataset to evaluate on"})
    data_dir: str = "./data"
    split: str = field(default="test", metadata={"help": "Dataset Split to evaluate on"})
    
    
    
def load_ir_evaluator(corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], 
                 qrels: Dict[str, Dict[str, int]], max_corpus_size: int = None, name: str = "eval", k_values: List[int] = [1, 3, 5, 10, 20, 100]) -> InformationRetrievalEvaluator:

    if len(queries) <= 0:
        raise ValueError("Dev Set Empty!, Cannot evaluate on Dev set.")

    rel_docs = {}
    corpus_ids = set()

    # need to convert corpus to cid => doc      
    corpus = {idx: corpus[idx].get("title") + " " + corpus[idx].get("text") for idx in corpus}

    # need to convert dev_qrels to qid => Set[cid]        
    for query_id, metadata in qrels.items():
        rel_docs[query_id] = set()
        for corpus_id, score in metadata.items():
            if score >= 1:
                corpus_ids.add(corpus_id)
                rel_docs[query_id].add(corpus_id)

    if max_corpus_size:
        # check if length of corpus_ids > max_corpus_size
        if len(corpus_ids) > max_corpus_size:
            raise ValueError("Your maximum corpus size should atleast contain {} corpus ids".format(len(corpus_ids)))
        new_corpus = {idx: corpus[idx] for idx in corpus_ids}
        for corpus_id in corpus_ids:
            corpus.pop(corpus_id, None)
        for corpus_id in random.sample(list(corpus), max_corpus_size - len(corpus_ids)):
            new_corpus[corpus_id] = corpus[corpus_id]
        corpus = new_corpus
    logger.info("{} set contains {} documents and {} queries".format(name, len(corpus), len(queries)))
    
    evaluator = InformationRetrievalEvaluator(
        queries = queries,
        corpus = corpus,
        relevant_docs = rel_docs,
        mrr_at_k = k_values, 
        ndcg_at_k = k_values,
        accuracy_at_k = k_values,
        precision_recall_at_k = k_values,
        map_at_k = k_values,
        show_progress_bar = True
    )
    
    return evaluator
    
    

def process_ir_retrieval_results(scores: Dict, evaluator) -> Dict[str, float]:
    results = {}
    keys = evaluator.csv_headers[2:]
    values = []
    k_values = evaluator.accuracy_at_k
    for name in evaluator.score_function_names:
        for k in k_values:
            values.append(scores[name]["accuracy@k"][k])
        for k in k_values:
            values.append(scores[name]["precision@k"][k])
            values.append(scores[name]["recall@k"][k])
        for k in k_values:
            values.append(scores[name]["mrr@k"][k])
        for k in k_values:
            values.append(scores[name]["ndcg@k"][k])
        for k in k_values:
            values.append(scores[name]["map@k"][k])
    return {k: v for k, v in zip(keys, values)}

    
    


def evaluate(eval_args: EvaluationArguments):  
    logger.info("Evaluating Model {} on dataset {} with split {}".format(eval_args.model_name_or_path, eval_args.dataset, eval_args.split))
    # Load Model
    model = load_model(eval_args.model_name_or_path)
    # Load Dataset
    dataset_args = BeirDatasetArguments(
        dataset = eval_args.dataset,
        data_dir = eval_args.data_dir
    )
    data_path = download_dataset(dataset_args)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=eval_args.split)
    evaluator = load_ir_evaluator(corpus, queries, qrels, name = eval_args.split)
    scores = evaluator.compute_metrices(model)
    results = process_ir_retrieval_results(scores, evaluator)
    results.update(asdict(eval_args))
    logger.info("Results")
    if wandb_logging:
        wandb.log(results)    
    print(results)

if __name__ == "__main__":
    
    
    try:
        import wandb
        wandb.init(project="prompt_tuning_information_retrieval", entity="ethankim10", tags=["eval"])
        wandb_logging = True
    except:
        pass

    parser = HfArgumentParser(
        EvaluationArguments
    )
    eval_args = parser.parse_args_into_dataclasses()[0]
    evaluate(eval_args)