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
python -m torch.distributed.launch --nproc_per_node=8
--model_name_or_path bert-base-uncased \
--train_n_passages 1 \
--output_dir ./models/base_model \
--report_to wandb \
--use_delta True \
--untie_encoder True \
--pooling mean \
--normalize True\
--train_dir ./data/pretraining/ 
--overwrite_output_dir True 
--save_steps 1000 \
--per_device_train_batch_size 16 \
```


## Train Dense Encoders for Each Domain 


