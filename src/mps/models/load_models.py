from .sentence_transformers_wrapper import DeltaModelSentenceTransformer
from .prompt_tuning import SoftPromptModelArguments, load_soft_prompt_model
from .utils import HFModelArguments
from pathlib import Path

from dataclasses import asdict
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Union, Dict

import logging

logger = logging.getLogger(name=__name__)


def wrap_soft_prompt_model_sentence_transformer(model_args):
    model, tokenizer = load_soft_prompt_model(model_args)
    model = DeltaModelSentenceTransformer(modules=[model], tokenizer=tokenizer)
    return model


def extract_model_args(cfg: Dict) -> SoftPromptModelArguments:
    args = SoftPromptModelArguments(
        model_name_or_path=cfg["model_name_or_path"],
        soft_prompt_token_number=cfg["soft_prompt_token_number"],
        init_from_vocab=cfg["init_from_vocab"],
        freeze_plm=cfg["freeze_plm"],
        tokenizer_name=cfg["tokenizer_name"],
    )
    return args


def load_model(
    model_name_or_path: str,
) -> Union[SentenceTransformer, DeltaModelSentenceTransformer]:
    model_path = Path("./models/trained_models").joinpath(model_name_or_path)
    embedding_path = model_path.joinpath("prompt_embeddings.npz")
    if embedding_path.is_file():
        embeddings = np.load(embedding_path)
        config_path = model_path.joinpath("config.json")
        config = json.load(open(config_path, "r"))
        model_args = extract_model_args(config)
        model = wrap_soft_prompt_model_sentence_transformer(model_args)
        model.load(model_path)
        model.config = asdict(model_args)
    else:
        model = SentenceTransformer(model_name_or_path)
    return model
