from sentence_transformers.util import batch_to_device
from sentence_transformers import SentenceTransformer
from torch import Tensor
from numpy import ndarray
import numpy as np
from tqdm import tqdm, trange
import torch
from typing import *

from pathlib import Path
import os

import logging
import json

logger = logging.getLogger(name=__name__)


class DeltaModelSentenceTransformer(SentenceTransformer):
    def __init__(self, modules: list, tokenizer, config: Dict = None):
        super(DeltaModelSentenceTransformer, self).__init__(modules=modules)
        self.soft_prompt_token_number = self.get_soft_token_parameters().shape[0]
        self.tokenizer = tokenizer
        self.config = config

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        return self.tokenizer(
            texts,
            max_length=200,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

    def forward(self, kwargs, for_train=True):
        output = self._first_module().forward(**kwargs)
        embeddings = torch.mean(
            output["last_hidden_state"][:, : self.soft_prompt_token_number, :], axis=1
        )
        #if for_train:
            #embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return {"sentence_embedding": embeddings}
        if for_train:
            if "pooler_output" not in output:
                embeddings = torch.mean(output["last_hidden_state"], axis=1)
                return {"sentence_embedding": embeddings}
            else:
                return {"sentence_embedding": output["pooler_output"]}
        else:
            if "pooler_output" not in output:
                embeddings = torch.mean(outputs["last_hidden_state"], axis=1)
                return {"sentence_embedding": embeddings}
            else:
                return {"sentence_embedding": output["pooler_output"]}

    def get_soft_token_parameters(self) -> torch.Tensor:
        return self.get_submodule(target="0.soft_prompt_layer").state_dict()[
            "soft_embeds"
        ]

    def load(self, path: str, **kwargs):
        if path is None:
            return
        embedding_path = Path(path).joinpath("prompt_embeddings.npz")
        with open(embedding_path, "rb") as f:
            embeddings = np.load(f)
        self.get_submodule(target="0.soft_prompt_layer").load_state_dict(
            {"soft_embeds": torch.tensor(embeddings)}
        )

    def save(self, path: str, **kwargs):
        if path is None:
            return
        os.makedirs(path, exist_ok=True)
        prompt_embeddings = self.get_soft_token_parameters().detach().cpu().numpy()
        embedding_output_path = Path(path).joinpath("prompt_embeddings.npz")
        logger.info("Saving Soft Prompt Embeddings")
        with open(embedding_output_path, "wb") as out:
            np.save(out, prompt_embeddings)
        config_path = Path(path).joinpath("config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f)

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "pooler_output",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings
        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.
        :return:
            By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = True

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self._target_device

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        output_value = "sentence_embedding"
        for start_index in trange(
            0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, device)
            with torch.no_grad():
                out_features = self.forward(features, for_train=False)

                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(
                        out_features[output_value], out_features["attention_mask"]
                    ):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0 : last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features["sentence_embedding"])):
                        row = {
                            name: out_features[name][sent_idx] for name in out_features
                        }
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(
                            embeddings, p=2, dim=1
                        )

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings


def wrap_soft_prompt_model_sentence_transformer(model_args):
    model, tokenizer = load_soft_prompt_model(model_args)
    model = DeltaModelSentenceTransformer(modules=[model], tokenizer=tokenizer)
    return model
