from torch.nn.functional import softmax
import torch
from dataclasses import dataclass, field, asdict

from transformers import HfArgumentParser
from src.mps.similarity.domain_similarity import DomainSimilarity
from src.mps.eval.eval_beir import evaluate, EvaluationArguments
from src.mps.datasets import DATASET_GROUPS

import numpy as np

import json
from typing import Dict

from pathlib import Path



import pandas as pd
import wandb


import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TransferEvaluationArguments:
    source_dataset_group: str = field(
        default="BEIR", metadata={"help": "set of source datasets"}
    )
    target_dataset: str = field(
        default=None, metadata={"help": "name of target dataset to evaluate on "}
    )
    top_k: int = field(
        default=5,
        metadata={
            "help": "number of top k most similar source datasts to retrieve from"
        },
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "temperature to use when weight averaging soft prompts"},
    )
    similarity_method: str = field(
        default="average",
        metadata={"help": "similarity metric to use for determining domain similarity"},
    )

    load_from_wandb: bool = field(
        default=True,
        metadata={"help": "whether to load prompts wandb artifacts"},
    )
    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={"help": "base model name"},
    )
    source_model_path: str = field(
        default="./models",
        metadata={
            "help": "path to folder with set of trained models, or tag to set of wandb model training runs"
        },
    )

    def __post__init__(self):
        # top k + # of source datasets
        if self.top_k > len(DATASET_GROUPS[self.source_dataset_list]):
            raise ValueError("Top k cannot be greate than number of source datasets")


def get_weights(scores, temperature: float = 1.0):
    domains = [_[0] for _ in scores]
    similarities = torch.tensor([_[1] for _ in scores])
    weights = softmax(similarities / temperature)
    weight_dict = {domain: weight.item() for domain, weight in zip(domains, weights)}
    return weight_dict


def get_weighted_prompts(
    weights: Dict[str, float], embeddings_dict: Dict[str, np.ndarray]
) -> np.ndarray:

    transfer_embeddings = np.zeros_like(list(embeddings_dict.values())[0])
    for domain, weight in weights.items():
        embeddings = embeddings_dict[domain]
        transfer_embeddings += embeddings * weight
    return transfer_embeddings
                                 
                                 
def load_wandb_embeddings(tag: str, project: str = "ir-transfer/prompt_tuning_information_retrieval") -> Dict[str, np.ndarray]:
    runs = api.runs(project)
    embedding_dict = {}
    for run in runs:
        if tag in run.tags:
            dataset = run.config["train_dataset"]
            run_path = project + "/" + run.id
            root_path = "./models/" + tag + dataset
            embedding_path = wandb.restore("prompt_embeddings.npz", run_path = run_path, root = root_path)
            embeddings = np.load(open(Path(root_path).joinpath("prompt_embeddings.npz"), "rb"))
            embedding_dict[dataset] = embeddings
    return embedding_dict



def eval_transfer(eval_args: TransferEvaluationArguments, wandb_logging: bool = False):
    logger.info("Running OOD transfer Experiment")
    source_datasets = DATASET_GROUPS[eval_args.source_dataset_group]
    runs = api.runs("ir-transfer/prompt_tuning_information_retrieval")
    source_datatsets = [run.config["train_dataset"] for run in runs if eval_args.source_dataset_group in run.tags]
    logger.info("Using {} source datasets from {}".format(len(source_datasets), eval_args.source_dataset_group))
    logger.info("Base Model is {}".format(eval_args.model_name_or_path))
    logger.info("Determining Domain Similarity with {} top k {} and temperature {}".format(eval_args.similarity_method, eval_args.top_k, eval_args.temperature))
    if eval_args.load_from_wandb:
        embedding_dict = load_wandb_embeddings(eval_args.source_model_path)
    else:
        embedding_dict = None
    domain_similarity = DomainSimilarity(
        domains=list(embedding_dict.keys()), method=eval_args.similarity_method
    )
    scores = domain_similarity.return_domain_similarities(eval_args.target_dataset, k = eval_args.top_k)
    weights = get_weights(scores, temperature=eval_args.temperature)
    prompt_embeddings = get_weighted_prompts(weights, embedding_dict)
    output_dir = Path("./models/trained_models").joinpath(
        eval_args.source_dataset_group
        + eval_args.similarity_method
        + eval_args.target_dataset
        #+ eval_args.temperature
        #+ str(eval_args.top_k)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_output_path = output_dir.joinpath("prompt_embeddings.npz")
    config_dict = asdict(eval_args)
    
    sample_run = [run for run in runs if eval_args.source_model_path in run.tags][0]
    config_dict.update({"model_name_or_path":eval_args.model_name_or_path})
    config_output_path = output_dir.joinpath("config.json")
    config_dict.update(sample_run.config)
    json.dump(config_dict, open(str(config_output_path), "w"))
    np.save(open(str(embedding_output_path), "wb"), prompt_embeddings)        
    eval_script_args = EvaluationArguments(
        model_name_or_path=str(output_dir), dataset=eval_args.target_dataset, split="test"
    )
    evaluate(eval_script_args, wandb_logging = wandb_logging)

if __name__ == "__main__":
    api = wandb.Api()            
    parser = HfArgumentParser(TransferEvaluationArguments)
    eval_args = parser.parse_args_into_dataclasses()[0]                    
    # try:
    import wandb
    wandb.init(
        project="prompt_tuning_information_retrieval",
        entity="ir-transfer",
        tags=["transfer_eval", eval_args.source_dataset_group + "transfer"],
    )
    wandb_logging = True
    # except:
    #     pass
    wandb.config.update(asdict(eval_args))
    eval_transfer(eval_args, wandb_logging = wandb_logging)
