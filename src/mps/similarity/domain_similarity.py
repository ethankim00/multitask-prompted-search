from typing import Dict, List, Tuple, Iterable
import numpy as np
from sentence_transformers import SentenceTransformer
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from abc import ABC
import json
from pathlib import Path

from src.mps.utils import download_dataset

from src.mps.datasets import CQA_DATASETS
from src.mps.datasets import TOP_LEVEL_OAG_DATASETS


class DomainFeatureExtractor(ABC):
    def extract_features():
        raise NotImplementedError


class SoftPromptEmbeddingFeatureExtractor(DomainFeatureExtractor):
    def extract_features(self, domains: List[str], data_dir: str = "./data/beir"):
        pass


class UnsupervisedDomainFeatureExtractor(DomainFeatureExtractor):
    def __init__(
        self, encoder_model: str = "all-MiniLM-L6-v2", data_dir: str = "./data/"
    ):

        self.encoder_model = SentenceTransformer(encoder_model)
        self.data_dir = data_dir

    def extract_features(self, domains: List[str]):
        result = {}
        for domain in domains:
            features = self._extract_features(domain)
            result[domain] = features

        return result

    def _extract_features(self, domain: str, data_dir: str = "./data/") -> np.ndarray:
        documents = self._load_documents(domain, "corpus.jsonl")
        queries = self._load_documents(domain, "queries.jsonl")
        texts = documents + queries
        return self._encode_texts(texts)

    def _load_documents(
        self, domain: str, file_name: str, num: int = 100
    ) -> Iterable[str]:
        if domain in CQA_DATASETS:
            domain = "cqadupstack" + "/" + domain
        elif domain in TOP_LEVEL_OAG_DATASETS:
            domain = "oag_qa_top_level" + "/" + domain
        folder_path = Path(self.data_dir).joinpath(domain)
        document_path = folder_path.joinpath(file_name)
        document_list = []
        with open(document_path, "r") as json_file:
            for i, line in enumerate(json_file):
                if i == num:
                    break
                document_list.append(json.loads(line))
        try:
            document_list = [
                document["title"] + " " + document["text"] for document in document_list
            ]
        except KeyError:
            document_list = [document["text"] for document in document_list]
        return document_list[0:num]

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.encoder_model.encode(texts)
        return embeddings


class DomainSimilarity:
    def __init__(
        self,
        method: str = "average",
        feature_extractor: DomainFeatureExtractor = UnsupervisedDomainFeatureExtractor(),
        domains: List[str] = None,
    ):
        self.method = method
        self.feature_extractor = feature_extractor
        self.domains = domains
        self.embeddings = self.get_domain_embeddings()
        self.similarities = self.calculate_pairwise_similarities(self.embeddings)

    def get_domain_embeddings(
        self,
    ):
        return self.feature_extractor.extract_features(self.domains)

    def calculate_pairwise_similarities(
        self,
        embeddings: Dict,
    ) -> np.ndarray:
        similarities = np.zeros((len(self.domains), len(self.domains)))
        for pair in combinations(list(embeddings.keys()), 2):
            similarity = self._calculate_similarities(pair, embeddings)
            i, j = self.domains.index(pair[0]), self.domains.index(pair[1])
            similarities[i, j] = similarity
        return similarities + similarities.T  # Fill in similarity Matrix

    def _calculate_similarities(
        self, pair: Tuple[str, str], embeddings: np.ndarray
    ) -> float:
        emb1, emb2 = embeddings[pair[0]], embeddings[pair[1]]
        if self.method == "average":
            return cosine_similarity(
                np.mean(emb1, axis=0).reshape(1, -1),
                np.mean(emb2, axis=0).reshape(1, -1),
            )
        elif self.method == "pairwise":
            return np.mean(cosine_similarity(emb1, emb2))

    def return_domain_similarities(
        self, domain: str, k: int = 5
    ) -> List[Tuple[str, float]]:
        domain_similarites = [
            (self.domains[i], score)
            for i, score in enumerate(self.similarities[self.domains.index(domain)])
            if self.domains[i] != domain
        ]
        if k is not None:
            print(k)
            domain_similarites.sort(key=lambda x: x[1], reverse=True)
            domain_similarites = domain_similarites[0:k]
        return domain_similarites


class ModelCombiner:
    pass


# return a the combined parameters of a soft prompt model with an option to actually initialize it
