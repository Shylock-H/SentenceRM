#!/bin/bash

set -e 
set -x

export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1

OUTPUT=$1
ZERO_STAGE=3
DATA_PATH='ultrafeedback_split_3l'

MODEL_NAME='meta-llama/Llama-3.1-8B'
# Your SFT model
MODEL_PATH=''


SEED=1234

if [ "$OUTPUT" == "" ]; then
    TIME_STEP=`date "+%Y-%m-%d-%H-%M-%S"`
    OUTPUT="./log_debug/sentence_reward_model-${MODEL_NAME/'/'/_}-$TIME_STEP-$SEED"
fi
mkdir -p $OUTPUT

deepspeed --master_port 12331 --include localhost:4,5,6,7 scripts/train_srm.py \
   --data_path $DATA_PATH \
   --data_output_path "/tmp/data_files/${MODEL_NAME/'/'/_}" \
   --data_split 0,10 \
   --model_name_or_path $MODEL_PATH \
   --per_device_train_batch_size 4 \
   --train_batch_size 256 \
   --max_seq_len 1024 \
   --learning_rate 3e-6 \
   --weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --seed $SEED \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --reward_embed_dim 256 \
   --weighted_sum \
   --penalty_coef 0.0 \
   --lora_rank 0 \
   --lora_alpha 8 \
   --bf16 \
   --offload \
   &> $OUTPUT/training.log
   
bash sh/run_reward_bench.sh $OUTPUT > $OUTPUT/eval.log