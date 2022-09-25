""" Train Models on BEIR datasets using the sentence transformers API"""
import numpy as np
import pandas as pd
import random
import wandb
import json
from src.mps.models import SoftPromptModelArguments, load_soft_prompt_model, DeltaModelSentenceTransformer
from src.mps.utils import  download_dataset, BeirDatasetArguments
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import logging
from transformers import HfArgumentParser
import os
from sentence_transformers import losses, models, SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import import_from_string, batch_to_device
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import logging

from pathlib import Path
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
from dataclasses import dataclass, field, asdict
import torch
from typing import *

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch import nn
from tqdm import trange

@dataclass
class TrainingArguments:
    model_name: str = None
    learning_rate: float = 3e-5
    batch_size: int = 32
    loss_function: str = "MNRL"
    weight_decay: float = 0
    num_epochs: int = 1
    wandb_log: bool = True
    prompt_tune: bool = True
    gradient_accumulation_steps: int = 1

class Trainer:
    
    
      def fit(self,
            model, 
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = 500,
            checkpoint_save_total_limit: int = 0,
            wandb_log: bool = False,
            gradient_accumulation_steps: int = 1,
        ):

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        model.to(model._target_device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = model.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(model._target_device)

        model.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = model._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)


        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data
                    labels = labels.to(model._target_device)
                    features = list(map(lambda batch: batch_to_device(batch, model._target_device), features))

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels)

                        if wandb_log:
                            wandb.log({"loss": loss_value.item()}, step = global_step)
                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        if _ % gradient_accumulation_steps == 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                            skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        if wandb_log:
                            wandb.log({"loss": loss_value.item()}, step = global_step)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()
                    if wandb_log:
                        lr = scheduler.get_last_lr()[-1]
                        wandb.log({"lr": scheduler.get_last_lr()[-1]}, step = global_step)

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1
                

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    model._eval_during_training(evaluator, output_path, save_best_model, epoch, global_step, callback)

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                    model._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

            if epoch % 10 ==0:
                logger.info("Evaluating")
                model._eval_during_training(evaluator, output_path, save_best_model, epoch, global_step, callback)

        if evaluator is None and output_path is not None:   #No evaluator, but output path: save final model version
            model.save(output_path)

        if checkpoint_path is not None:
            model._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)
            
            
      
    
def training_callback(score: float, epoch: int, steps: int):
    wandb.log({
        "score": score,
        "epoch": epoch,
    }, step = steps)
        

def validate_data_splits(data_path: str):
    data_path = Path(data_path)
    base_path = data_path.joinpath("qrels")
    train_path = base_path.joinpath("train.tsv")
    dev_path = base_path.joinpath("dev.tsv")
    test_path = base_path.joinpath("test.tsv")

    paths = [train_path, dev_path, test_path]

    if all([path.is_file() for path in paths]):
        return

    if test_path.is_file():
        if not any([train_path.is_file(), dev_path.is_file()]):
            df = pd.read_csv(test_path, delimiter="\t")
            train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df)), int(.9*len(df))])
            train.to_csv(train_path, sep = "\t", index = False)
            validate.to_csv(dev_path, sep = "\t", index = False)
            test.to_csv(test_path, sep = "\t", index = False)
            return

    if not train_path.is_file():
        test_qrels = pd.read_csv(data_path.joinpath("qrels").joinpath("test.tsv"))
        
    
def load_ir_evaluator( corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], 
                 qrels: Dict[str, Dict[str, int]], max_corpus_size: int = None, name: str = "eval") -> SentenceEvaluator:
    # TODO move to utils
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

        # Add mandatory corpus documents
        new_corpus = {idx: corpus[idx] for idx in corpus_ids}

        # Remove mandatory corpus documents from original corpus
        for corpus_id in corpus_ids:
            corpus.pop(corpus_id, None)

        # Sample randomly remaining corpus documents
        for corpus_id in random.sample(list(corpus), max_corpus_size - len(corpus_ids)):
            new_corpus[corpus_id] = corpus[corpus_id]

        corpus = new_corpus

    logger.info("{} set contains {} documents and {} queries".format(name, len(corpus), len(queries)))
    return InformationRetrievalEvaluator(queries, corpus, rel_docs, name=name, show_progress_bar = True)
    
    
    
    
def train(
    dataset_args: BeirDatasetArguments,
    training_args: TrainingArguments,
    model_args: SoftPromptModelArguments,
):  
    data_path = download_dataset(dataset_args)
    if training_args.prompt_tune:
        model, tokenizer= load_soft_prompt_model(model_args)
        model = DeltaModelSentenceTransformer(modules = [model], tokenizer = tokenizer)
    else:
        model = SentenceTransformer(model_args.model_name_or_path)
    retriever = TrainRetriever(model=model, batch_size=training_args.batch_size)
    validate_data_splits(data_path)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")
    max_corpus_size = min(len(corpus), 100000)
    logger.info("Corpus Size: {}".format(max_corpus_size))
    try:
        dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")
        ir_evaluator = load_ir_evaluator(dev_corpus, dev_queries, dev_qrels, max_corpus_size = max_corpus_size)
    except:
        logger.info("No dev set found for evaluation, loading dummy evaluator")
        ir_evaluator = retriever.load_dummy_evaluator()
    train_samples = retriever.load_train(corpus, queries, qrels)
    train_dataloader = retriever.prepare_train(train_samples, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
    model_save_path = os.path.join(
        "models",
        "trained_models",
        "{}-v1-{}-lr-{}-bs-{}-lf{}-ep-{}-wd-{}-spt-{}".format(
            model_args.model_name_or_path,
            dataset_args.dataset,
            training_args.learning_rate,
            training_args.batch_size,
            training_args.loss_function,
            training_args.num_epochs,
            training_args.weight_decay,
            model_args.soft_prompt_token_number,
        ),
    )
    os.makedirs(model_save_path, exist_ok=True)
    
    evaluation_steps = 800
    warmup_steps = int(
        len(train_samples) * training_args.num_epochs / retriever.batch_size * 0.05
    )

    trainer = Trainer()
    callback = training_callback #if training_args.wandb_log else None
    if training_args.wandb_log:
        if training_args.prompt_tune:
            wandb.watch(model.get_submodule(target = '0.soft_prompt_layer'), log_freq = 1)
    trainer.fit(
        model = model, 
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=ir_evaluator,
        epochs=training_args.num_epochs,
        optimizer_params={
            "lr": training_args.learning_rate,
            "eps": 1e-6,
        },
        weight_decay = training_args.weight_decay,
        output_path=model_save_path,
        warmup_steps=warmup_steps,
        evaluation_steps=evaluation_steps,
        save_best_model=True,
        use_amp=True,
        callback = callback,
        wandb_log = training_args.wandb_log,
        gradient_accumulation_steps = training_args.gradient_accumulation_steps,
        checkpoint_path = model_save_path,
        checkpoint_save_steps = 1000,
        checkpoint_save_total_limit = 50, 
        
    )
    logger.info("Saving model info")
    model_params = asdict(training_args)
    model_params["train_dataset"] = dataset_args.dataset
    model_params.update(asdict(model_args))
    with open(Path(model_save_path).joinpath("config.json"), "w") as f:
        json.dump(model_params, f)
            


if __name__ == "__main__":
    logger = logging.getLogger("BEIR_training")
    parser = HfArgumentParser(
        [BeirDatasetArguments, TrainingArguments, SoftPromptModelArguments]
    )
    dataset_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    if training_args.wandb_log:
        import wandb
        wandb.init(project="prompt_tuning_information_retrieval", entity="ethankim10", tags=["train"])
        wandb_logging = True
        model_params = asdict(training_args)
        model_params["train_dataset"] = dataset_args.dataset
        model_params.update(asdict(model_args))
        wandb.config.update(model_params)
    
    
    train(dataset_args, training_args, model_args)
