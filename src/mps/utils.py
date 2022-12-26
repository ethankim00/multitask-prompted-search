from dataclasses import dataclass, field
from src.mps.datasets import OAGBeirConverter, BEIR_DATASETS, OAG_DATASETS, CQA_DATASETS
from beir import util, LoggingHandler

import pandas as pd
import os
from pathlib import Path


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BeirDatasetArguments:
    dataset: str = field(default=None, metadata={"help": "Beir Dataset to train on"})
    data_dir: str = "./data"


def download_dataset(dataset_args: BeirDatasetArguments) -> str:
    if dataset_args.dataset in BEIR_DATASETS:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
            dataset_args.dataset
        )
        out_dir = os.path.join(Path("./", dataset_args.data_dir))
        data_dir = util.download_and_unzip(url, out_dir)
    elif dataset_args.dataset in OAG_DATASETS:
        converter = OAGBeirConverter(
            data_dir=Path(dataset_args.data_dir).joinpath("oag_qa")
        )
        data_dir = converter.convert(dataset_args.dataset)
    elif dataset_args.dataset in CQA_DATASETS:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip"
        out_dir = os.path.join(Path("./", dataset_args.data_dir))
        data_dir = util.download_and_unzip(url, out_dir)
        data_dir = data_dir + "/" + dataset_args.dataset
    # elif dataset_args.dataset in TOP_LEVEL_OAG:
    #     s    pass
    # TODO logic for top level oag topics
    # Convert all lower level datasets
    # Compbine in a single directory, concatenative files and adjusting index mappings
    # 1 load
    return data_dir


def validate_data_splits(data_path: str):
    """
    Checks if the data splits are already present in the data_path. If not, it creates them.

    Args:
        data_path (str): Path to the data directory
    """
    data_path = Path(data_path)
    base_path = data_path.joinpath("qrels")
    train_path = base_path.joinpath("train.tsv")
    dev_path = base_path.joinpath("dev.tsv")
    test_path = base_path.joinpath("test.tsv")

    paths = [train_path, dev_path, test_path]

    if all([path.is_file() for path in paths]):
        train_df = pd.read_csv(train_path, delimiter="\t")
        if len(train_df) > 10000:  # limit to 10000 training examples
            logger.info("Limiting Train Dataset")
            train_df = train_df[0:10000]
            train_df.to_csv(train_path, sep="\t", index=False)
        return

    if test_path.is_file():
        if not any([train_path.is_file(), dev_path.is_file()]):
            logger.info("Creating training and validation splits from test dataset")
            df = pd.read_csv(test_path, delimiter="\t")
            train, validate, test = np.split(
                df.sample(frac=1, random_state=42),
                [int(0.8 * len(df)), int(0.9 * len(df))],
            )
            train.to_csv(train_path, sep="\t", index=False)
            validate.to_csv(dev_path, sep="\t", index=False)
            test.to_csv(test_path, sep="\t", index=False)
            return

        if train_path.is_file() and not dev_path.is_file():
            logger.info("Creating validation split from train dataset")
            df = pd.read_csv(
                data_path.joinpath("qrels").joinpath("train.tsv"), delimiter="\t"
            )
            train, validate = np.split(
                df.sample(frac=1, random_state=42), [int(0.8 * len(df))]
            )
            train.to_csv(train_path, sep="\t", index=False)
            validate.to_csv(dev_path, sep="\t", index=False)
            return

        if not train_path.is_file() and dev_path.is_file():
            logger.info("Creating training split from test and dev datasets")
            test_df = pd.read_csv(
                data_path.joinpath("qrels").joinpath("test.tsv"), delimiter="\t"
            )
            val_df = pd.read_csv(
                data_path.joinpath("qrels").joinpath("dev.tsv"), delimiter="\t"
            )
            df = pd.concat([test_df, val_df])
            train, validate, test = np.split(
                df.sample(frac=1, random_state=42),
                [int(0.8 * len(df)), int(0.9 * len(df))],
            )
            train.to_csv(train_path, sep="\t", index=False)
            validate.to_csv(dev_path, sep="\t", index=False)
            test.to_csv(test_path, sep="\t", index=False)
            return
