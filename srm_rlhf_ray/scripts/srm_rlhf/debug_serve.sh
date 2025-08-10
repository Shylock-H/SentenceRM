set -x

export CUDA_VISIBLE_DEVICES=1

source TODO

time_start=$(date +%s)
save_path=TODO

python -m openrlhf.cli.serve_test \
    --reward_pretrain TODO \
    --port 5000 \
    --bf16 \
    --flash_attn \
    --normalize_reward \
    --max_len 1024 \
    --batch_size 4 > ${save_path}/serve_sparse_srm_${time_start}.log 2>&1


time_start=$(date +%s)

python -m openrlhf.cli.serve_test \
    --reward_pretrain TODO \
    --port 5001 \
    --bf16 \
    --flash_attn \
    --normalize_reward \
    --max_len 1024 \
    --batch_size 4 > ${save_path}/serve_sparse_srm_${time_start}.log 2>&1
