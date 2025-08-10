#!/bin/bash

set -e 
set -x

export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=3
# Your dataset obtained from preprocessing
DATA_PATH='' 
MODEL_NAME='meta-llama/Llama-3.1-8B'

SEED=1234

if [ "$OUTPUT" == "" ]; then
    TIME_STEP=`date "+%Y-%m-%d-%H-%M-%S"`
    OUTPUT="./log_sft/model-${MODEL_NAME/'/'/_}-$TIME_STEP-$SEED"
fi
mkdir -p $OUTPUT

deepspeed --master_port 12344 --include localhost:4,5,6,7 scripts/train_sft.py \
   --data_path $DATA_PATH \
   --data_output_path "/tmp/data_files/${MODEL_NAME/'/'/_}" \
   --data_split 10,0 \
   --model_name_or_path $MODEL_NAME \
   --per_device_train_batch_size 4 \
   --train_batch_size 256 \
   --max_seq_len 1024 \
   --learning_rate 3e-6 \
   --weight_decay 0.2 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine_with_min_lr \
   --num_warmup_steps 0 \
   --seed $SEED \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --lora_rank 0 \
   --lora_alpha 8 \
   --bf16 \
   --offload \
   &> $OUTPUT/training.log