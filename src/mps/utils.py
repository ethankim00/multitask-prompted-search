from dataclasses import dataclass, field
from src.mps.datasets import OAGBeirConverter, BEIR_DATASETS, OAG_DATASETS, CQA_DATASETS
from beir import util, LoggingHandler

import os
from pathlib import Path


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
