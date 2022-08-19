""" Train Models on BEIR datasets using the sentence transformers API"""

from src.mps.models import SoftPromptModelArguments, load_soft_prompt_model, DeltaModelSentenceTransformer
import logging
from transformers import HfArgumentParser
import os
from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import logging

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field


@dataclass
class BeirDatasetArguments:
    dataset: str = field(default=None, metadata={"help": "Beir Dataset to train on"})
    data_path: str = "./data"


@dataclass
class TrainingArugments:
    model_name: str = None
    learning_rate: float = 3e-5
    batch_size: int = 32
    loss_function: str = "MNRL"
    num_epochs: int = 1


def download_dataset(dataset_args: BeirDatasetArguments):
    
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset_args.dataset
    )
    out_dir = os.path.join(pathlib.Path("./", dataset_args.data_path))
    data_path = util.download_and_unzip(url, out_dir)
    return data_path


#https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip

def train(
    dataset_args: BeirDatasetArguments,
    training_args: TrainingArugments,
    model_args: SoftPromptModelArguments,
):
    data_path = download_dataset(dataset_args)
    model, tokenizer= load_soft_prompt_model(model_args)
    model = DeltaModelSentenceTransformer(modules = [model], tokenizer = tokenizer)
    retriever = TrainRetriever(model=model, batch_size=training_args.batch_size)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")
    #### Please Note not all datasets contain a dev split, comment out the line if such the case
    dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")
    train_samples = retriever.load_train(corpus, queries, qrels)
    train_dataloader = retriever.prepare_train(train_samples, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
    ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)

    #### If no dev set is present from above use dummy evaluator
    # ir_evaluator = retriever.load_dummy_evaluator()

    #### Provide model save path
    model_save_path = os.path.join(
        "models",
        "output",
        "{}-v1-{}".format(model_args.model_name_or_path, dataset_args.dataset),
    )
    os.makedirs(model_save_path, exist_ok=True)
    
    evaluation_steps = 2000
    warmup_steps = int(
        len(train_samples) * training_args.num_epochs / retriever.batch_size * 0.1
    )

    retriever.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=ir_evaluator,
        epochs=training_args.num_epochs,
        optimizer_params={
            "lr": training_args.learning_rate,
            "eps": 1e-6,
            "correct_bias": False,
        },
        output_path=model_save_path,
        warmup_steps=warmup_steps,
        evaluation_steps=evaluation_steps,
        save_best_model=False,
        use_amp=True,
    )


if __name__ == "__main__":
    logger = logging.getLogger("BEIR_training")
    parser = HfArgumentParser(
        [BeirDatasetArguments, TrainingArugments, SoftPromptModelArguments]
    )
    dataset_args, training_args, model_args = parser.parse_args_into_dataclasses()
    train(dataset_args, training_args, model_args)
