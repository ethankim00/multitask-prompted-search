import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from shutil import copytree
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import HfArgumentParser

import wandb
from openmatch.arguments import InferenceArguments as EncodingArguments
from src.mps.datasets import DATASET_GROUPS, DATASET_GROUPS_MAPPING
from src.mps.similarity.domain_similarity import DomainSimilarity
from src.mps.utils import BEIRDataArguments, download_dataset, validate_data_splits
from src.mps.models.prompt_tuning.prompt_tuning_model import PromptModelArguments

from .eval import eval_beir

logging.basicConfig(level=logging.INFO)
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
    weights: Dict[str, float],
    embeddings_dict: Union[
        Dict[str, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]
    ],
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    # deal w/ untied embeddings where values in the embedding dict are tuples of query and passage embeddings
    if isinstance(list(embeddings_dict.values())[0], tuple):
        query_transfer_embeddings = np.zeros_like(list(embeddings_dict.values())[0][0])
        passage_transfer_embeddings = np.zeros_like(
            list(embeddings_dict.values())[0][1]
        )
        for domain, weight in weights.items():
            query_embeddings, passage_embeddings = embeddings_dict[domain]
            query_transfer_embeddings += query_embeddings * weight
            passage_transfer_embeddings += passage_embeddings * weight
        return query_transfer_embeddings, passage_transfer_embeddings
    else:
        transfer_embeddings = np.zeros_like(list(embeddings_dict.values())[0])
        for domain, weight in weights.items():
            embeddings = embeddings_dict[domain]
            transfer_embeddings += embeddings * weight
        return transfer_embeddings


def load_wandb_embeddings(
    tag: str, project: str = "ir-transfer/prompt_tuning_information_retrieval"
) -> Dict[str, np.ndarray]:

    api = wandb.Api()
    runs = api.runs(project)
    embedding_dict = {}
    for run in runs:
        if tag in run.tags:
            dataset = run.config["train_dataset"]
            if dataset is not None:
                artifact_id = run.id
                try:
                    artifact = api.artifact(project + "/" + artifact_id + ":v0")
                except:
                    logger.warning("No trained model found for source dataset {}".format(dataset))
                artifact_dir = artifact.download("./models/transfer/" + dataset)
                open_match_config = json.load(
                    open(Path(artifact_dir).joinpath("openmatch_config.json"), "r")
                )
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if open_match_config["tied"]:
                    embeddings = torch.load(
                        open(Path(artifact_dir).joinpath("query_model").joinpath("pytorch_model.bin"), "rb"),
                        map_location=device,
                    )["soft_prompt_layer.soft_embeds"].cpu().numpy()
                    embedding_dict[dataset] = embeddings
                else:
                    query_embeddings = torch.load(
                        open(
                            Path(artifact_dir)
                            .joinpath("query_model")
                            .joinpath("pytorch_model.bin"),
                            "rb",
                        ),
                        map_location=device,
                    )["soft_prompt_layer.soft_embeds"].cpu().numpy()
                    passage_embeddings = torch.load(
                        open(
                            Path(artifact_dir)
                            .joinpath("passage_model")
                            .joinpath("pytorch_model.bin"),
                            "rb",
                        ),
                        map_location=device,
                    )["soft_prompt_layer.soft_embeds"].cpu().numpy()
                    embedding_dict[dataset] = (query_embeddings, passage_embeddings)
  
    return embedding_dict


def eval_transfer(eval_args: TransferEvaluationArguments):
    logger.info("Running OOD transfer Experiment")
    source_datasets = DATASET_GROUPS[eval_args.source_dataset_group]
    runs = api.runs("ir-transfer/prompt_tuning_information_retrieval")
    source_datasets = [
        run.config["train_dataset"]
        for run in runs
        if eval_args.source_dataset_group in run.tags
    ]
    for dataset in source_datasets:
        data_path = download_dataset(dataset)
        validate_data_splits(data_path)
    logger.info(
        "Using {} source datasets from {}".format(
            len(source_datasets), eval_args.source_dataset_group
        )
    )
    logger.info("Base Model is {}".format(eval_args.model_name_or_path))
    logger.info(
        "Determining Domain Similarity with {} top k {} and temperature {}".format(
            eval_args.similarity_method, eval_args.top_k, eval_args.temperature
        )
    )
    if eval_args.load_from_wandb:
        embedding_dict = load_wandb_embeddings(eval_args.source_dataset_group)
    else:
        embedding_dict = None
    domains=list(embedding_dict.keys())
    if eval_args.target_dataset not in domains:
        domains.append(eval_args.target_dataset)
    domain_similarity = DomainSimilarity(
        domains=domains, method=eval_args.similarity_method
    )
    scores = domain_similarity.return_domain_similarities(
        eval_args.target_dataset, k=eval_args.top_k
    )
    weights = get_weights(scores, temperature=eval_args.temperature)
    if os.getenv("WANDB_DISABLED") != "True":
        wandb.log({"weights": weights})
    prompt_embeddings = get_weighted_prompts(weights, embedding_dict)
    output_dir = Path("./models/").joinpath(
        eval_args.source_dataset_group
        + eval_args.similarity_method
        + eval_args.target_dataset
        + str(eval_args.temperature)
        + str(eval_args.top_k)
    )
    # output_dir.mkdir(parents=True, exist_ok=True)
    source_model_path = "models/transfer/" + source_datasets[0]

    copytree(source_model_path, output_dir)
    if isinstance(prompt_embeddings, tuple):
        query_embeddings, passage_embeddings = prompt_embeddings
        torch.save(
            {"soft_prompt_layer.soft_embeds": torch.from_numpy(query_embeddings)},
            output_dir.joinpath("query_model").joinpath("pytorch_model.bin"),
        )
        torch.save(
            {"soft_prompt_layer.soft_embeds": torch.from_numpy(passage_embeddings)},
            output_dir.joinpath("passage_model").joinpath("pytorch_model.bin"),
        )
    else:
        torch.save(
            {"soft_prompt_layer.soft_embeds": torch.from_numpy(prompt_embeddings)},
            output_dir.joinpath("pytorch_model.bin"),
        )

    # # Run the evaluation
    # Construct the evaluation arguments
    model_args = PromptModelArguments(model_name_or_path=output_dir, pooling="mean", normalize =False)
    data_args = BEIRDataArguments(
        eval_dataset=eval_args.target_dataset,
        doc_template="<title> [SEP] <text>",
        query_template="<text>",
    )
    encoding_args = EncodingArguments(output_dir = os.path.join(output_dir,"eval"))
    eval_beir(model_args=model_args, data_args=data_args, encoding_args=encoding_args)


if __name__ == "__main__":
    api = wandb.Api()
    parser = HfArgumentParser(TransferEvaluationArguments)
    eval_args = parser.parse_args_into_dataclasses()[0]
    # Conditional wandb logging
    if os.getenv("WANDB_DISABLED") != "True":
        import wandb

        wandb.init(
            project="prompt_tuning_information_retrieval",
            entity="ir-transfer",
            tags=["transfer_eval", eval_args.source_dataset_group + "transfer"],
        )
        wandb.config.update(asdict(eval_args))
    eval_transfer(eval_args)
