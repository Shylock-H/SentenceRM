set -x

source TODO

time_start=$(date +%s)
save_path=TODO

fuser -k 5000/tcp

export CUDA_VISIBLE_DEVICES=7
python -m openrlhf.cli.serve_dense_srm \
    --reward_pretrain TODO \
    --port 5000 \
    --bf16 \
    --flash_attn \
    --normalize_reward \
    --max_len 1024 \
    --batch_size 16

wait
