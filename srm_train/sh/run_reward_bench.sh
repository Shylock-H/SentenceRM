set -e 
set -x

export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1

# model_path='/home/liyc/workspace/project/SentenceRM/log/reward_model-Qwen_Qwen2.5-7B-2024-12-16-11-25-34-1234'
model_path=$1

deepspeed --master_port 14434 --include localhost:7 eval/reward_bench.py \
          --path $model_path
