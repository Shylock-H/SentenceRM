# Usage

## Environment Setup

Python version: 3.11.11
Torch version: 2.6.0+cu124

```bash
pip install -r requirements.txt
pip install -e .
```

## Dataset Preprocessing

Replace the `cache_dir="TODO"` with the path to your cache directory.

```bash
python scripts/datasets/down.py
```

This script downloads the `nz/anthropic-hh-golden-rlhf` dataset for training.

## Training Process

### 1. Start Ray Cluster

First, start the Ray cluster which is used for distributed training:

```bash
bash scripts/srm_rlhf/ray_start.sh
```

This starts a Ray cluster with a memory usage threshold of 0.99 and allocates GPUs.

### 2. Serve SRM

Replace `TODO` in the script with:
- Your environment path
- The path to your reward model

```bash
bash scripts/srm_rlhf/serve_dense_srm.sh
```

This script will:
- Serve the reward model
- Run on port 5000
- Use bf16 precision and flash attention
- Normalize rewards
- Set maximum length to 1024
- Use batch size 16

### 3. Train Policy

After serving the reward model, train your policy with:

```bash
bash scripts/srm_rlhf/train_dense_actor.sh
```

This script:
- Sets up a distributed training environment
- Saves code snapshots for reproducibility
- Uses Ray for distributed execution
- Takes advantage of vLLM for efficient inference
- Logs training metrics via Weights & Biases

### 4. Inference

To evaluate your trained model, use:

```bash
bash scripts/srm_rlhf/inference.sh
```

Replace the following values in the script:
- `SRM_RLHF_WORK_DIR`: Path to your working directory
- `data_path` and `data_name`: Evaluation dataset details
- `model_path_list`: Path to your trained model(s)

The script will generate outputs for your evaluation dataset.





