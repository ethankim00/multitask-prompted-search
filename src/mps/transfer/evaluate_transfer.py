
from typing import List, Union


from dataclasses import dataclass, field
from src.mps.eval.eval_beir import evaluate, EvaluationArguments
from src.mps.similarity.domain_similarity import DomainSimilarity
from src.mps.datasets import DATASET_GROUPS

@dataclass
class TransferArguments:
    source_datasets: Union[List[str],str] =  field(
        metadata={
            "help": "List of source datsets or string representing predefined group of source datasets"
        }
    )
    target_dataset: str 
    temperature: float = 0.1
    top_k: int = 4
    similarity_metric: str = "average"
    model_dir: str = "./models"
    
    
def get_weighted_prompts(weights : Dict[str, float], model_path_suffix: str = None) -> np.ndarray
    transfer_embeddings = np.zeros((40, 768))
    for domain, weight in weights.items():
        
        # TODO find better way to load models from folders
        base_path = Path("./models/trained_models")
        if domain in ["scifact", "nfcorpus", "scidocs"]:
            model_path = "bert-base-uncased-v1-" + domain + "-lr-0.005-bs-32-lfMNRL-ep-100-wd-0-spt-40"
        else:
            model_path = "bert-base-uncased-v1-" + domain + "-lr-0.005-bs-32-lfMNRL-ep-40-wd-0-spt-40"
        model_path = base_path.joinpath(model_path)
        embeddings = np.load(model_path.joinpath("prompt_embeddings.npz"))
        transfer_embeddings  += embeddings * weight
    return transfer_embeddings
    
    
def get_weights(scores, temperature: float = 1.0):
    domains = [_[0] for _ in scores]
    similarities = torch.tensor([_[1] for _ in scores])
    weights = softmax(similarities/temperature)
    weight_dict = {domain: weight.item() for domain, weight in zip(domains, weights)}
    return weight_dict
    
def evaluate_transfer(transfer_args: TransferArguments):
    if isinstance(transfer_args.source_datasets, str):
        transfer_args.source_datasets = DATASET_GROUPS[transfer_args.source_datasets]
    domain_similarity = DomainSimilarity(domains = transfer_args.source_datasets, method =transfer_args.similarity_metric)
    scores = domain_similarity.return_domain_similarities(domain=transfer_args.target_dataset, k = transfer_args.top_k)
    weights = get_weights(scores, temperature=transfer_args.temperature)
    prompt_embeddings = get_weighted_prompts(weights)
    # Load model with correct config and eval on it on correct dataset
    #Load config from one of the source model folders
    # create new folder with transfer experiment details
    # save config with only the relevant keys
    # pass the path to new folder to eval 
    
    eval_args = EvaluationArguments(model_name_or_path = , dataset = transfer_args.target_dataset, )