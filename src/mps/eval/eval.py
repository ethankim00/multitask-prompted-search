import logging
import os
import sys
import wandb
import json
import random
from dataclasses import asdict
import pytrec_eval
from openmatch.arguments import DataArguments
from openmatch.arguments import InferenceArguments as EncodingArguments
from openmatch.arguments import ModelArguments
from openmatch.dataset import BEIRDataset
from openmatch.modeling import DRModelForInference
from openmatch.retriever import Retriever
from openmatch.utils import save_as_trec
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser


from src.mps.models.prompt_tuning.prompt_tuning_model import (
    PromptDRInferenceModel,
    PromptModelArguments,
)
from src.mps.utils import construct_beir_dataset, BEIRDataArguments
from src.mps.datasets import DATASET_GROUPS_MAPPING


logger = logging.getLogger(__name__)


def eval_beir(
    model_args: PromptModelArguments,
    data_args: BEIRDataArguments,
    encoding_args: EncodingArguments,
):

    if os.path.exists(encoding_args.output_dir) and os.listdir(
        encoding_args.output_dir
    ):
        if not encoding_args.overwrite_output_dir:
            raise ValueError(
                f"Output directory ({encoding_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        else:
            # remove all files in the output directory
            if encoding_args.local_process_index == 0:
                for file in os.listdir(encoding_args.output_dir):
                    os.remove(os.path.join(encoding_args.output_dir, file))

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if encoding_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed inference: %s, 16-bits inference: %s",
        encoding_args.local_rank,
        encoding_args.device,
        encoding_args.n_gpu,
        bool(encoding_args.local_rank != -1),
        encoding_args.fp16,
    )
    logger.info("Encoding parameters %s", encoding_args)
    logger.info("MODEL parameters %s", model_args)

    # num_labels = 1
    # config = AutoConfig.from_pretrained(
    #     model_args.config_name
    #     if model_args.config_name
    #     else model_args.model_name_or_path,
    #     num_labels=num_labels,
    #     cache_dir=model_args.cache_dir,
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    model = PromptDRInferenceModel.build(
        model_args=model_args,
        # config=config,
        cache_dir=model_args.cache_dir,
    )

    if data_args.eval_dataset is not None:
        eval_dir = construct_beir_dataset(
            data_args.eval_dataset, tokenizer=tokenizer, split="test"
        )
        data_args.data_dir = eval_dir
    corpus_dataset_path = os.path.join(data_args.data_dir, "corpus.jsonl")
    num_lines = sum(1 for line in open(corpus_dataset_path))
    if num_lines > 100000:
        logger.info("Corpus dataset is greater than 100K documents")
        # load the all the document ids from the train test and dev qrels
        def get_all_qrel_doc_ids(qrels_path) -> list:
            qrel_doc_ids = []
            with open(qrels_path, "r") as f:
                # skip the header
                for i, line in enumerate(f):
                    if i == 0:
                        continue
                    doc_id = line.split("\t")[1]
                    qrel_doc_ids.append(doc_id)
            return qrel_doc_ids

        all_qrel_doc_ids = []
        for split in ["train", "test", "dev"]:
            qrels_path = os.path.join(data_args.data_dir, "qrels", f"{split}.tsv")
            all_qrel_doc_ids.extend(get_all_qrel_doc_ids(qrels_path))
        corpus_dataset = {}
        with open(corpus_dataset_path, "r") as f:
            for line in f:
                doc = json.loads(line)
                corpus_dataset[doc["_id"]] = doc
        new_corpus_dataset = []
        all_qrel_doc_ids = list(set(all_qrel_doc_ids))
        for doc_id in all_qrel_doc_ids:
            new_corpus_dataset.append(
                {
                    "_id": doc_id,
                    "text": corpus_dataset[doc_id]["text"],
                    "title": corpus_dataset[doc_id]["title"],
                }
            )
            del corpus_dataset[doc_id]
        random.seed(42)
        random_sample = random.sample(
            list(corpus_dataset.keys()), 100000 - len(new_corpus_dataset)
        )
        for doc_id in random_sample:
            new_corpus_dataset.append(
                {
                    "_id": doc_id,
                    "text": corpus_dataset[doc_id]["text"],
                    "title": corpus_dataset[doc_id]["title"],
                }
            )
        os.remove(corpus_dataset_path)
        with open(corpus_dataset_path, "w") as f:
            for doc in new_corpus_dataset:
                f.write(json.dumps(doc) + "\n")
        logger.info("Corpus dataset limited to 100K documents")

    beir_dataset = BEIRDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        stream=True,
        batch_size=encoding_args.per_device_eval_batch_size,
        num_processes=encoding_args.world_size,
        process_index=encoding_args.process_index,
        cache_dir=model_args.cache_dir,
    )

    if encoding_args.use_split_search:
        retriever = Retriever.build_embeddings(
            model, beir_dataset.corpus_dataset, encoding_args
        )
        run = retriever.split_retrieve(
            beir_dataset.query_datasets["test"], topk=encoding_args.retrieve_depth
        )
    else:
        retriever = Retriever.build_all(
            model, beir_dataset.corpus_dataset, encoding_args
        )
        run = retriever.retrieve(
            beir_dataset.query_datasets["test"], topk=encoding_args.retrieve_depth
        )

    if encoding_args.local_process_index == 0:

        # Save trec file
        if encoding_args.trec_save_path is None:
            encoding_args.trec_save_path = os.path.join(
                encoding_args.output_dir, "test.trec"
            )
        save_as_trec(run, encoding_args.trec_save_path)

        # compute metric
        metrics = pytrec_eval.supported_measures
        metrics.add("success.20")
        evaluator = pytrec_eval.RelevanceEvaluator(beir_dataset.qrels["test"], metrics)
        eval_results = evaluator.evaluate(run)

        def print_line(measure, scope, value):
            print("{:25s}{:8s}{:.4f}".format(measure, scope, value))
            with open(
                os.path.join(encoding_args.output_dir, "test_result.log"),
                "w",
                encoding="utf-8",
            ) as fw:
                fw.write("{:25s}{:8s}{:.4f}\n".format(measure, scope, value))

        for query_id, query_measures in sorted(eval_results.items()):
            for measure, value in sorted(query_measures.items()):
                pass

        # Scope hack: use query_measures of last item in previous loop to
        # figure out all unique measure names.
        #
        # TODO(cvangysel): add member to RelevanceEvaluator
        #                  with a list of measure names.
        results = {}
        for measure in sorted(query_measures.keys()):
            value = pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure] for query_measures in eval_results.values()],
            )
            print_line(
                measure,
                "all",
                value,
            )
            results[measure] = value
        if os.getenv("WANDB_DISABLED") != "True":
            tags = ["eval"]
            tags += DATASET_GROUPS_MAPPING[data_args.eval_dataset]
            wandb.init(
                project="prompt_tuning_information_retrieval",
                entity="ir-transfer",
                tags=tags,
            )
            output_dict = {}
            for args in [model_args, data_args, encoding_args]:
                output_dict.update(asdict(args))
            output_dict.update(results)
            print(output_dict)
            wandb.config.update(output_dict, allow_val_change=True)
            wandb.log(results)


if __name__ == "__main__":
    parser = HfArgumentParser(
        (PromptModelArguments, BEIRDataArguments, EncodingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, encoding_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, encoding_args = parser.parse_args_into_dataclasses()
        model_args: PromptModelArguments
        data_args: BEIRDataArguments
        encoding_args: EncodingArguments
    eval_beir(data_args=data_args, model_args=model_args, encoding_args=encoding_args)
