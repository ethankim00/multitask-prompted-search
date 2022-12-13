## Soft Prompt Transfer for Information Retrieval

## Install Requirements

```
conda install -c pytorch faiss-gpu cudatoolkit=11.0
conda install -c conda-forge cudatoolkit=11.0.3
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install Openmatch
https://github.com/OpenMatch/OpenMatch
```
git clone https://github.com/OpenMatch/OpenMatch.git
cd OpenMatch
pip install -e .
```
Optionally install Gradcache
```
git clone https://github.com/luyug/GradCache
cd GradCache
pip install .
```


## Perform Multitask Pretraining on Dense Encoder

### Download DPR

### Convert to Openmatch Format 


### Train the dense Encoder


## Train Dense Encoders for Each Domain 


