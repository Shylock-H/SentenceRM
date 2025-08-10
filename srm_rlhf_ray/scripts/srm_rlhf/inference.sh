#!/bin/bash
set -x
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export SRM_RLHF_WORK_DIR=TODO
# data_path=tatsu-lab/alpaca_eval
# data_name=alpaca_eval

data_path=TODO
data_name=TODO

model_path_list=(
    TODO
)

for model_path in "${model_path_list[@]}"; do
    echo "========================================"
    echo "Processing model: ${model_path##*/}"
    
    if [[ ! -d "$model_path" ]]; then
        echo "[WARNING] model_path $model_path does not exist, skipping..."
        continue
    fi
    output_path=${model_path}/arena-hard.json
    echo "output_path: $output_path"
    
    read -r -d '' inference_commands <<EOF
openrlhf.cli.batch_inference \
   --eval_task generate \
   --pretrain $model_path \
   --dataset $data_path \
   --dataset_split eval \
   --output_path $output_path \
   --max_new_tokens 512 \
   --temperature 1.0 \
   --micro_batch_size 64 \
   --input_key prompt \
   --label_key prompt
EOF

    if [[ ${1} != "slurm" ]]; then
        echo "Starting inference with deepspeed..."
        deepspeed --master_port 12578 --include localhost:2 --module $inference_commands
    fi
    
    echo "Finished processing: ${model_path##*/}"
    echo "========================================"
    echo
done