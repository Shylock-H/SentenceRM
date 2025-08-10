import torch
import deepspeed.comm as dist
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4


def get_train_ds_config(
    offload,
    stage=2,
    enable_hybrid_engine=False,
    inference_tp_size=1,
    release_inference_cache=False,
    pin_parameters=True,
    tp_gather_partition_size=8,
    max_out_tokens=512,
    enable_tensorboard=False,
    enable_mixed_precision_lora=False,
    bf16=False,
    memory_efficient_linear=False,
    tb_path="",
    tb_name="",
):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device},
        "offload_optimizer": {"device": device},
        "stage3_param_persistence_threshold": 1e5,  # (1e4,1e6)
        "stage3_max_live_parameters": 1e8,  # (3e7, 1e9)
        "stage3_prefetch_bucket_size": 1e8,  # (3e7, 5e8)
        "memory_efficient_linear": memory_efficient_linear,
    }
    if enable_mixed_precision_lora:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        if dist.get_world_size() != torch.cuda.device_count():
            zero_opt_dict["zero_hpz_partition_size"] = torch.cuda.device_count()
    ds_config = {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard",
        },
    }
    if bf16:
        ds_config["bf16"] = {
            "enabled": True,
        }
    else:
        ds_config["fp16"] = {
            "enabled": True,
            "loss_scale_window": 100,  # 100
        }
    return ds_config


def get_eval_ds_config(offload, stage=0, bf16=False, memory_efficient_linear=False):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {"device": device},
        "memory_efficient_linear": memory_efficient_linear,
    }
    ds_config = {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }
    if bf16:
        ds_config["bf16"] = {
            "enabled": True,
        }
    else:
        ds_config["fp16"] = {
            "enabled": True,
        }
    return ds_config

def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    no_decay_name_list=["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters

def _z3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]