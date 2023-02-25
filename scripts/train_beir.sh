# Train in domain encoders for each bier domain 
# set hyperparameters
MODEL_NAME_OR_PATH=./models/dpr_pretrain_1_hard_neg_bs_128_40ep/checkpoint-17000/
TRAIN_N_PASSAGES=1
OUTPUT_DIR=./models
REPORT_TO=wandb
USE_DELTA=True
UNTIE_ENCODER=False
POOLING=mean
NORMALIZE=False
OVERWRITE_OUTPUT_DIR=True
CUDA_AVAILABLE_DEVICES=0,1,2,3,4,5,6,7
NUM_TRAIN_EPOCHS=40
PER_DEVICE_TRAIN_BATCH_SIZE=128
LEARNING_RATE=1e-3
#webis-touche2020 dbpedia-entity
# touche, trec-covid fiqa
# arguana nq hotpotqa msmarco nfcorpus quora dbpedia scidocs fever climate-fever scifact android english gaming gis mathematica physics programmers stats tex unix webmasters wordpress arguana nq hotpotqa msmarco nfcorpus quora dbpedia scidocs fever climate-fever scifact android english gaming gis mathematica
for train_dataset in  physics programmers stats tex unix webmasters wordpress
do
    python -m src.mps.training.train \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --train_dataset $train_dataset \
        --train_n_passages $TRAIN_N_PASSAGES \
        --output_dir $OUTPUT_DIR/$train_dastaset \
        --report_to $REPORT_TO \
        --use_delta $USE_DELTA \
        --untie_encoder $UNTIE_ENCODER \
        --pooling $POOLING \
        --normalize $NORMALIZE \
        --overwrite_output_dir $OVERWRITE_OUTPUT_DIR \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --num_train_epochs $NUM_TRAIN_EPOCHS \
        --logging_steps 20 \
        --fp16 \
        --q_max_len 64 \
        --p_max_len 192 \
        --soft_prompt_token_number 50
done