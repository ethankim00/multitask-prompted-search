from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    HfArgumentParser,
)

import copy

import os
import torch
from torch import nn, Tensor
import numpy as np
import importlib
import json
from sklearn.metrics.pairwise import cosine_similarity

from typing import Optional, Dict, Union, Tuple, List
from dataclasses import dataclass, field

from src.mps.models.utils import HFModelArguments
import logging

logger = logging.getLogger(name=__name__)

from opendelta import SoftPromptModel

from openmatch.arguments import ModelArguments
from openmatch.modeling import DRModel
from openmatch.arguments import DataArguments
from openmatch.arguments import DRTrainingArguments as TrainingArguments
from openmatch.arguments import ModelArguments


from opendelta import SoftPromptModel
from opendelta import AutoDeltaModel
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import (
    AutoConfig,
    AutoModel,
    BatchEncoding,
    PreTrainedModel,
    T5EncoderModel,
)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


@dataclass
class PromptModelArguments(ModelArguments):
    use_delta: bool = True
    soft_prompt_token_number: int = 40
    init_from_vocab: bool = True
    freeze_plm: bool = True


from transformers.modeling_outputs import ModelOutput


@dataclass
class DROutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None


class PromptDRModel(DRModel):
    def __init__(
        self,
        lm_q: PreTrainedModel,
        lm_p: PreTrainedModel,
        q_delta_model=None,
        p_delta_model=None,
        tied: bool = True,
        feature: str = "last_hidden_state",
        pooling: str = "first",
        head_q: nn.Module = None,
        head_p: nn.Module = None,
        normalize: bool = False,
        model_args: ModelArguments = None,
        data_args: DataArguments = None,
        train_args: TrainingArguments = None,
    ):
        super(PromptDRModel, self).__init__(lm_q=lm_q, lm_p=lm_p)

        self.tied = tied
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.q_delta_model = q_delta_model
        self.p_delta_model = p_delta_model
        self.head_q = head_q
        self.head_p = head_p

        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

        self.feature = feature
        self.pooling = pooling
        self.normalize = normalize

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        if train_args is not None and train_args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError(
                    "Distributed training has not been initialized for representation all gather."
                )
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def save(self, output_dir: str):
        if not self.tied:
            os.makedirs(os.path.join(output_dir, "query_model"))
            os.makedirs(os.path.join(output_dir, "passage_model"))
            self.q_delta_model.save_finetuned(os.path.join(output_dir, "query_model"))
            self.p_delta_model.save_finetuned(os.path.join(output_dir, "passage_model"))
            if self.head_q is not None:
                self.head_q.save(os.path.join(output_dir, "query_head"))
                self.head_p.save(os.path.join(output_dir, "passage_head"))
        else:
            self.q_delta_model.save_finetuned(os.path.join(output_dir, "query_model"))
            if self.head_q is not None:
                self.head_q.save(output_dir)
        with open(os.path.join(output_dir, "openmatch_config.json"), "w") as f:
            json.dump(self._get_config_dict(), f, indent=4)
        if not os.getenv("WANDB_DISABLED"):
            import wandb

            logger.info("Uploading Checkpoint to Wandb")
            if not self.tied:
                wandb.save(
                    os.path.join(output_dir, "query_model/*"),
                    base_path="./",
                    policy="now",
                )
                wandb.save(
                    os.path.join(output_dir, "passage_model/*"),
                    base_path="./",
                    policy="now",
                )
                query_vocab = self.get_nearest_neighbor_vocabulary()
                passage_vocab = self.get_nearest_neighbor_vocabulary()
                table = wandb.Table(
                    columns=[
                        "soft_prompt_token_nearest_neighbors_query",
                        "soft_prompt_token_nearest_neighbors_passage",
                    ]
                )
                for i in range(len(query_vocab)):
                    table.add_data(query_vocab[i], passage_vocab[i])
                wandb.log({"soft_prompt_token_nearest_neighbors": table})
            else:

                wandb.save(
                    os.path.join(output_dir, "query_model/*"),
                    base_path="./",
                    policy="now",
                )
                # query_vocab = self.get_nearest_neighbor_vocabulary()
                # table = wandb.Table(columns=["soft_prompt_token_nearest_neighbors"])
                # for i in range(len(query_vocab)):
                #     table.add_data(query_vocab[i])
                # wandb.log({"soft_prompt_token_nearest_neighbors": table})

    def get_soft_token_parameters(
        self,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.tied:
            query_embeddings = self.q_delta_model.state_dict()["delta_modules.0.soft_embeds"].detach().cpu().numpy()
            return query_embeddings
        else:
            query_embeddings = self.q_delta_model.state_dict()["delta_modules.0.soft_embeds"].detach().cpu().numpy()
            passage_embeddings = self.p_delta_model.state_dict()["delta_modules.0.soft_embeds"].detach().cpu().numpy()
            return query_embeddings, passage_embeddings

    def get_nearest_neighbor_vocabulary(self, top_k: int = 20) -> List[str]:
        if self.tied:
            query_embeddings = self.get_soft_token_parameters()
            query_word_embeddings = (
                self.lm_q.base_model.embeddings.word_embeddings.weight#.detach()
                # .cpu()
                # .numpy()
            )
            #query_embeddings = query_embeddings.detach().cpu().numpy()
            similarities = cosine_similarity(query_word_embeddings, query_embeddings)
            top = np.argsort(similarities, axis=0)[-top_k:, :]
            return self.tokenizer.batch_decode(top.T)
        else:
            query_embeddings, passage_embeddings = self.get_soft_token_parameters()
            passage_word_embeddings = (
                self.lm_p.base_model.embeddings.word_embeddings.weight.detach()
                .cpu()
                .numpy()
            )
            query_word_embeddings = (
                self.lm_q.base_model.embeddings.word_embeddings.weight.detach()
                .cpu()
                .numpy()
            )
            query_similarities = cosine_similarity(
                query_word_embeddings, query_embeddings
            )
            passage_similarities = cosine_similarity(
                passage_word_embeddings, passage_embeddings
            )
            top_query = np.argsort(query_similarities, axis=0)[-top_k:, :]
            top_passage = np.argsort(passage_similarities, axis=0)[-top_k:, :]
            return self.tokenizer.batch_decode(
                top_query.T
            ), self.tokenizer.batch_decode(top_passage.T)

    def encode(self, items, model, head):
        if items is None:
            return None, None
        items = BatchEncoding(items)
        if "T5" in type(model).__name__ and not self.model_args.encoder_only:
            decoder_input_ids = torch.zeros(
                (items.input_ids.shape[0], 1), dtype=torch.long
            ).to(items.input_ids.device)
            items_out = model(
                **items, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            hidden = items_out.last_hidden_state
            reps = hidden[:, 0, :]
        else:
            items_out = model(**items, return_dict=True)
            hidden = getattr(items_out, self.feature)
            if self.feature == "pooler_output":
                return hidden, hidden
            if self.pooling == "first":
                reps = hidden[:, 0, :]
            elif self.pooling == "mean":
                soft_prompt_attention_mask = items.attention_mask
                soft_prompt_attention_mask[
                    :, : model.soft_prompt_token_number
                ] = torch.zeros(
                    (items.attention_mask.shape[0], model.soft_prompt_token_number)
                )
                reps = mean_pooling(
                    hidden, soft_prompt_attention_mask
                )  # only pool hidden reps of real tokens
            elif self.pooling == "no":
                reps = hidden
            else:
                raise ValueError("Unknown pooling type: {}".format(self.pooling))
        if head is not None:
            reps = head(reps)  # D * d
        if self.normalize:
            reps = F.normalize(reps, dim=1)
        return hidden, reps

    @staticmethod
    def _load_delta_model(model_args, model_type: str, **hf_kwargs):
        with open(
            os.path.join(model_args.model_name_or_path, model_type, "config.json")
        ) as f:
            config = json.load(f)
        model_name = config["backbone_checkpoint_name"]
        # model_class = getattr(importlib.import_module("transformers"), model_name)
        lm = AutoModel.from_pretrained(
            model_name, **hf_kwargs
        )  # TODO: have to fix if we ever train a non backbone model
        lm.soft_prompt_token_number = model_args.soft_prompt_token_number
        # delta_model = SoftPromptModel(
        #     lm,
        #     token_init=model_args.init_from_vocab,
        #     soft_token_num=model_args.soft_prompt_token_number,
        # )
        delta_model =  AutoDeltaModel.from_finetuned(
            os.path.join(model_args.model_name_or_path, model_type),
            lm,
            local_files_only=True,
        )
        del lm.embeddings.token_type_ids
        return lm, delta_model

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        data_args: DataArguments = None,
        train_args: TrainingArguments = None,
        **hf_kwargs,
    ):
        # load local
        config = None
        head_q = head_p = None
        q_delta_model = p_delta_model = None
        if os.path.exists(
            os.path.join(model_args.model_name_or_path, "openmatch_config.json")
        ):
            with open(
                os.path.join(model_args.model_name_or_path, "openmatch_config.json")
            ) as f:
                config = json.load(f)
        if (
            os.path.isdir(model_args.model_name_or_path) and config is not None
        ):  # an OpenMatch model load from directory, still want to load the backbone model from hub or from cache
            tied = config["tied"]
            if tied:
                lm_q, q_delta_model = cls._load_delta_model(
                    model_args, model_type="query_model"
                )
                # TODO support linear dimension reduction head
                lm_p, p_delta_model = lm_q, q_delta_model
                if config["linear_head"]:
                    head_q = head_p = LinearHead.load(model_args.model_name_or_path)
                if model_args.freeze_plm:
                    q_delta_model.freeze_module(
                        exclude=["deltas"], set_state_dict=True
                    )
            else:
                lm_q, q_delta_model = cls._load_delta_model(
                    model_args, model_type="query_model"
                )
                lm_p, p_delta_model = cls._load_delta_model(
                    model_args, model_type="passage_model"
                )
                if model_args.freeze_plm:
                    p_delta_model.freeze_module(
                        exclude=["deltas"], set_state_dict=True
                    )
                    q_delta_model.freeze_module(
                        exclude=["deltas"], set_state_dict=True
                    )
                if config["linear_head"]:
                    head_q = LinearHead.load(_qry_head_path)
                    head_p = LinearHead.load(_psg_head_path)
        else:  # a Huggingface model,we are loading from the first time, logic already handled
            tied = not model_args.untie_encoder
            model_class = T5EncoderModel if model_args.encoder_only else AutoModel
            lm_q = model_class.from_pretrained(
                model_args.model_name_or_path, **hf_kwargs
            )
            if model_args.use_delta:
                q_delta_model = SoftPromptModel(
                    lm_q,
                    token_init=model_args.init_from_vocab,
                    soft_token_num=model_args.soft_prompt_token_number,
                )
                lm_q.soft_prompt_token_number = model_args.soft_prompt_token_number
                if model_args.freeze_plm:
                    q_delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
                    q_delta_model.log()
                if model_args.untie_encoder:
                    lm_p = model_class.from_pretrained(
                        model_args.model_name_or_path, **hf_kwargs
                    )
                    p_delta_model = SoftPromptModel(
                        lm_p,
                        token_init=model_args.init_from_vocab,
                        soft_token_num=model_args.soft_prompt_token_number,
                    )
                    lm_p.soft_prompt_token_number = model_args.soft_prompt_token_number
                    if model_args.freeze_plm:
                        p_delta_model.freeze_module(
                            exclude=["deltas"], set_state_dict=True
                        )
                else:
                    lm_p = lm_q

            # TODO support linear dimension reduction head
            lm_p = copy.deepcopy(lm_q) if not tied else lm_q
            if model_args.add_linear_head:
                head_q = LinearHead(
                    model_args.projection_in_dim, model_args.projection_out_dim
                )
                head_p = copy.deepcopy(head_q) if not tied else head_q

        model = cls(
            tied=tied,
            lm_q=lm_q,
            lm_p=lm_p,
            q_delta_model=q_delta_model,
            p_delta_model=p_delta_model,
            feature=model_args.feature
            if config is None
            else config["plm_backbone"]["feature"],
            pooling=model_args.pooling if config is None else config["pooling"],
            head_q=head_q,
            head_p=head_p,
            normalize=model_args.normalize if config is None else config["normalize"],
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
        )
        return model


class PromptDRInferenceModel(PromptDRModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval()

    @torch.no_grad()
    def encode_passage(self, psg):
        return super(PromptDRInferenceModel, self).encode_passage(psg)

    @torch.no_grad()
    def encode_query(self, qry):
        return super(PromptDRInferenceModel, self).encode_query(qry)

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
        q_hidden, q_reps = self.encode_query(query)
        p_hidden, p_reps = self.encode_passage(passage)
        return DROutput(q_reps=q_reps, p_reps=p_reps)
