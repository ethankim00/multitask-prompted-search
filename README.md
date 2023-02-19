## Soft Prompt Transfer for Information Retrieval

## Install Requirements

```
conda create -n prompt python=3.9
conda activate prompt
conda install pytorch==1.12.1 cudatoolkit=11.4 pytorch transformers faiss-gpu -c conda-forge
pip install -r requirements.txt 
pip install sentence-transformers --no-dependencies
pip install nltk
pip install -U scikit-learn --no-dependencies
pip install -U threadpoolctl --no-dependencies
pip install -U PIL --no-dependencies
pip install -U Pillow--no-dependencies
pip install -U Pillow --no-dependencies
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

```
sh scripts/download_dpr_data.sh
```

### Convert to Openmatch Format 

```
sh scripts/process_pretraining_data.sh
```
### Multitask Pretrain the base Model

Train on the multitask mixture of 4 datasets from the DPR paper. Follow the recommended hyperparameters and train for 40 epochs with bs = 128

```
python -m src.mps.training.train 
--model_name_or_path bert-base-uncased \
--train_n_passages 2 \
--output_dir ./models/base_model \
--use_mapping_dataset False \
--report_to wandb \
--use_delta True \
--untie_encoder True \
--pooling mean \
--normalize True \
--train_dir ./data/pretraining/ 
--overwrite_output_dir True 
--save_steps 1000 \
--per_device_train_batch_size 128 \
--learning_rate 1e-3 \
--num_train_epochs 40 \
```


## Train Dense Encoders for Each Domain 

### BEIR Datasets
Run the script to train the dense encoder for each publicly available BEIR dataset. Note the subcategories of the CQADupStack dataset are not treated as separate datasets. 

```
sh scripts/train_beir.sh
```

### OAG QA Datasets 

#### Download OAG-QA Data

Download OAG-QA from the link: OAG-QA, and unzip it to ./data/oag_qa 

Run the script to train the dense encoder for each OAG-QA dataset

```
sh scripts/train_oag_qa.sh
```

