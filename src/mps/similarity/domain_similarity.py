from typing import Dict, List, Tuple, Iterable
import numpy as np
from sentence_transformers import SentenceTransformer
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from abc import ABC


class DomainFeatureExtractor(ABC):
    def extract_features():
        raise NotImplementedError


class SoftPromptEmbeddingFeatureExtractor(DomainFeatureExtractor):
    def extract_features(self, domains: List[str], data_dir: str = "./data/beir"):
        pass


class UnsupervisedDomainFeatureExtractor(DomainFeatureExtractor):
    def __init__(
        self, encoder_model: str = "all-MiniLM-L6-v2", data_dir: str = "./data/beir"
    ):

        self.encoder_model = SentenceTransformer(encoder_model)
        self.data_dir = data_dir

    def extract_features(self, domains: List[str]):
        result = {}
        for domain in domains:
            features = self._extract_features(domain)
            result[domain] = features

    def _extract_features(
        self, domain: str, data_dir: str = "./data/beir"
    ) -> np.ndarray:
        documents = self._load_documents(domain)
        queries = self._load_queries(domain)
        texts = documents + queries
        return self._encode_texts(texts)

    def _load_documents(self, domain: str, num: int = 100) -> Iterable[str]:
        pass

    def _load_queries(self, domain: str, num: int = 100) -> Iterable[str]:

        folder_path = Path(self.data_dir).joinpath("domain")
        query_path = folder_path.joinpath()
        document_path = folder_path.joinpath()

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
        for pair in combinations(list(embeddings.keys()), 2):
            similarity = self._calculate_similarities(pair, embeddings)

    def _calculate_similarities(
        self, pair: Tuple[str, str], embeddings: np.ndarray
    ) -> float:
        emb1, emb2 = embeddings[pair[0]], embeddings[pair[1]]
        if self.method == "average":
            return cosine_similarity(np.mean(emb1, axis=0), np.mean(emb2, axis=0))
        elif self.method == "pairwise":
            return np.mean(cosine_similarity(emb1, emb2))

    def return_most_similar_domains(
        self, domain: str, k: int = 5
    ) -> List[Tuple[str, float]]:
        pass


class ModelCombiner:
    pass


# return a the combined parameters of a soft prompt model with an option to actually initialize it
