# Train in domain encoders for each bier domain 
# set hyperparameters
MODEL_NAME_OR_PATH=bert-base-uncased
TRAIN_N_PASSAGES=1
OUTPUT_DIR=./models
REPORT_TO=wandb
USE_DELTA=True
UNTIE_ENCODER=True
POOLING=mean
NORMALIZE=True
OVERWRITE_OUTPUT_DIR=True
CUDA_AVAILABLE_DEVICES=0,1,2,3,4,5,6,7
NUM_TRAIN_EPOCHS=100
PER_DEVICE_TRAIN_BATCH_SIZE=16\
LEARNING_RATE=3e-3
for train_dataset in GEOMETRY STATISTICS ALGEBRA CALCULUS NUMBER_THEORY LINEAR_ALGEBRA ASTROPHYSICS QUANTUM_MECHANICS PHYSICS CHEMISTRY BIOCHEMISTRY HEALTH_CARE NATURAL_SCIENCE PSYCHOLOGY ALGORITHM NEURAL_NETWORK COMPUTER_VISION DATA_MINING DEEP_LEARNING MACHINE_LEARNING NLP ECONOMICS
do
    python -m torch.distributed.launch --nproc_per_node=8
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --train_dastaset $train_dataset \
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
        --negatives_x_device True
done