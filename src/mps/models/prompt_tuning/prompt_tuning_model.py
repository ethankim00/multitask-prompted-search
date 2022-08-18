from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    HfArgumentParser,
)

from typing import Optional
from dataclasses import dataclass, field

from mps.models.utils import HFModelArguments
import logging

logger = logging.getLogger(name=__name__)

from opendelta import SoftPromptModel

## load a huggingface BERT style model and wrap with soft prompt model


@dataclass
class SoftPromptModelArguments(HFModelArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    soft_prompt_token_number: int = field(
        default=20, metadata={"help": "Number of soft prompt embeddings to train"}
    )

    init_from_vocab: bool = field(
        default=True, metadata={"help": "Initialize soft prompt embeddings from vocab"}
    )
    freeze_plm: bool = field(
        default=True, metadata={"help": "Whether to freeze the LM backbone"}
    )


def load_soft_prompt_model(model_args: SoftPromptModelArguments):
    base_model = AutoModel.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )
    if model_args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer, cache_dir=model_args.cache_dir
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir
        )
    soft_prompt_model = SoftPromptModel(
        backbone_model=base_model,
        token_init=model_args.init_from_vocab,
        soft_token_num=model_args.soft_prompt_token_number,
    )
    if model_args.freeze_plm:
        soft_prompt_model.freeze_module()
    return base_model, tokenizer
