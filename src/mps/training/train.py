# Adapted from Tevatron (https://github.com/texttron/tevatron)

import logging
import os
import sys

from transformers import BatchEncoding

from openmatch.arguments import DataArguments
from openmatch.arguments import DRTrainingArguments as TrainingArguments
from openmatch.arguments import ModelArguments
from openmatch.dataset import QPCollator, StreamDRTrainDataset,MappingDRTrainDataset , StreamDREvalDataset
from openmatch.modeling import DRModel
from openmatch.trainer import DRTrainer as Trainer
from openmatch.trainer import GCDenseTrainer
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
#from transformers.integrations import TensorBoardCallback
from torch import Tensor
import torch.nn.functional as F
import torch

from opendelta import SoftPromptModel

logger = logging.getLogger(__name__)
from typing import *

from dataclasses import dataclass

@dataclass
class PromptModelArguments(ModelArguments):
    soft_prompt_token_number: int = 40
    init_from_vocab: bool = True
    freeze_plm : bool = False
    
from transformers.modeling_outputs import ModelOutput
@dataclass
class DROutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None
    
    
    
# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    
class PromptDRModel(DRModel):
    
    
    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
        q_hidden, q_reps = self.encode_query(query)
        p_hidden, p_reps = self.encode_passage(passage)

        if q_reps is None or p_reps is None:
            return DROutput(q_reps=q_reps, p_reps=p_reps)

        # if self.training:
        if self.train_args.negatives_x_device:
            q_reps = self.dist_gather_tensor(q_reps)
            p_reps = self.dist_gather_tensor(p_reps)

        effective_bsz = (
            self.train_args.per_device_train_batch_size * self.world_size
            if self.train_args.negatives_x_device
            else self.train_args.per_device_train_batch_size
        )
        scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * self.data_args.train_n_passages
        print(self.loss_fn)
        loss = self.loss_fn(scores, target)
        loss.backward()

        if self.training and self.train_args.negatives_x_device:
            loss = loss * self.world_size  # counter average weight reduction
        return DROutput(loss=loss, scores=scores, q_reps=q_reps, p_reps=p_reps)
    
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
            print(hidden.shape)
            if self.pooling == "first":
                reps = hidden[:, 0, :]
            elif self.pooling == "mean":
                soft_prompt_attention_mask = items.attention_mask
                soft_prompt_attention_mask[:, :model.soft_prompt_token_number] = torch.zeros((items.attention_mask.shape[0], model.soft_prompt_token_number))
                reps = mean_pooling(hidden, soft_prompt_attention_mask) # only pool hidden reps of real tokens
            elif self.pooling == "no":
                reps = hidden
            else:
                raise ValueError("Unknown pooling type: {}".format(self.pooling))
        if head is not None:
            reps = head(reps)  # D * d
        if self.normalize:
            reps = F.normalize(reps, dim=1)
        return hidden, reps
    


def train():
    parser = HfArgumentParser((PromptModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = PromptDRModel.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    delta_model = SoftPromptModel(model.lm_q,
        token_init=model_args.init_from_vocab,
        soft_token_num=model_args.soft_prompt_token_number)
    model.lm_q.soft_prompt_token_number = model_args.soft_prompt_token_number
    if model_args.freeze_plm:
        delta_model.freeze_module(exclude=["deltas"],set_state_dict=True)
        delta_model.log()
    if model_args.untie_encoder:
        delta_model = SoftPromptModel(model.lm_q,
        token_init=model_args.init_from_vocab,
        soft_token_num=model_args.soft_prompt_token_number)
        model.lm_p.soft_prompt_token_number = model_args.soft_prompt_token_number
        if model_args.freeze_plm:
            delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
    else:
        model.lm_p = model.lm_q
        print(model.lm_p)
    TrainDatasetClass = MappingDRTrainDataset if training_args.use_mapping_dataset else StreamDRTrainDataset
    train_dataset = TrainDatasetClass(tokenizer, data_args, shuffle_seed=training_args.seed, cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    eval_dataset = StreamDREvalDataset(tokenizer, data_args, cache_dir=data_args.data_cache_dir or model_args.cache_dir) if data_args.eval_path is not None else None

    #tb_callback = TensorBoardCallback()

    trainer_cls = GCDenseTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=QPCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
        #callbacks=[tb_callback]
    )
    train_dataset.trainer = trainer

    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
