from torch.nn.functional import softmax
import torch
from dataclasses import dataclass, field

from transformers import HfArgumentParser
from src.mps.similarity.domain_similarity import DomainSimilarity
from src.mps.eval.eval_beir import evaluate, EvaluationArguments
from src.mps.datasets import DATASET_GROUPS

import numpy as np
from typing import Dict

field(
    default="mean",
    metadata={"help": "similarity metric to use for determining domain similarity"},
)


import pandas as pd
import wandb

api = wandb.Api()


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
    similarity_method: float = field(
        default="average",
        metadata={"help": "similarity metric to use for determining domain similarity"},
    )

    load_from_wandb: bool = field(
        default=True,
        metadata={"help": "whether to load prompts wandb artifacts"},
    )

    source_model_path: str = field(
        default="./models",
        metadata={
            "help": "path to folder with set of trained models, or path to set of wandb model training runs"
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

    transfer_embeddings = np.zeros_like(list(embeddings_dict.values()[0]))
    for domain, weight in weights.items():
        embeddings = embeddings_dict[domain]
        transfer_embeddings += embeddings * weight
    return transfer_embeddings


def eval_transfer(eval_args: TransferEvaluationArguments):

    source_datasets = DATASET_GROUPS[eval_args.source_dataset_group]

    # Load modelss
    # 1. Load locally

    # 2. Load artifacts from wandb api
    if eval_args.load_from_wandb:

        for dataset in source_datasets:
            run_path = eval_args.sour
        a = wandb.restore(
            "prompt_embeddings.npz",
            run_path="ir-transfer/prompt_tuning_information_retrieval/1u0z6gtm",
            root="./models",
        )

        pass

    ## Calculate Similarities
    embeddings_dict = None
    domain_similarity = DomainSimilarity(
        domains=source_datasets, method=eval_args.similarity_method
    )
    scores = domain_similarity.return_domain_similarities(eval_args.target_dataset)

    weights = get_weights(scores, temperature=eval_args.temperature)
    prompt_embeddings = get_weighted_prompts(weights, embeddings_dict)

    # Calculate embeddings with weights

    ## Save model allong with config:

    ## cal eval on this new model


if __name__ == "__main__":
    try:
        import wandb

        wandb.init(
            project="prompt_tuning_information_retrieval",
            entity="ethankim10",
            tags=["transfer_eval"],
        )
        wandb_logging = True
    except:
        pass

    parser = HfArgumentParser(TransferEvaluationArguments)
    eval_args = parser.parse_args_into_dataclasses()[0]
    eval_transfer(eval_args)
