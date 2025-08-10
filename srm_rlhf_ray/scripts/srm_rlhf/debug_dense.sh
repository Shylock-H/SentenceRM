set +x 

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export SRM_RLHF_WORK_DIR="TODO"
export WANDB_API_KEY="8c93813de025e8a0c964310fc76adca78d85be40"

save_path="${SRM_RLHF_WORK_DIR}/checkpoint/SRM-Sparse-Llama3.1-Tulu-3-8B-$(date +'%Y%m%d-%H%M%S')"
wandb_name="SRM-Sparse-Llama3.1-Tulu-3-8B-$(date +'%Y%m%d-%H%M%S')"

if [[ ! -d $save_path ]]; then
    mkdir -p $save_path
fi

ray job submit --address="http://127.0.0.1:8266" \
   --runtime-env-json='{
      "working_dir": "TODO",
      "excludes": [
         "**/model_weights/**",
         "**/datasets/**",
         "**/checkpoint/**",
         "**/hub/**",
         "**weights/**"
      ],
      "env_vars": {
         "HF_ENDPOINT": "https://hf-mirror.com",
         "CUDA_VISIBLE_DEVICES": "0",
         "RAY_DEBUG": "1"
      }}' \
   -- python3 -m openrlhf.cli.train_dense_srm_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.4 \
   --colocate_actor_ref \
   --pretrain ${SRM_RLHF_WORK_DIR}/hub/Llama-3.1-Tulu-3-8B-SFT \
   --remote_rm_url http://localhost:5000/get_reward \
   --save_path $save_path \
   --ckpt_path $save_path \
   --wandb_run_name $wandb_name \
   --wandb_project SRM_RLHF \
   --micro_train_batch_size 1 \
   --train_batch_size 2 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 4 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 512 \
   --generate_max_len 512 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data ${SRM_RLHF_WORK_DIR}/datasets/ultrafeedback-ppo \
   --advantage_estimator reinforce \
   --apply_chat_template \
   --input_key prompt \
   --label_key prompt \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
   --use_wandb $WANDB_API_KEY \
   --n_samples_per_prompt 1 \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --save_hf_ckpt \
   --disable_ds_ckpt
