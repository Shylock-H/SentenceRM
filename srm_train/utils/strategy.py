# modified from https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/deepspeed.py
import os
import math
import random
import shutil
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from typing import List, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from peft import PeftModel, get_peft_model_state_dict
from torch import distributed as dist
from torch.optim import Optimizer

from transformers import PreTrainedTokenizer, default_data_collator
from model import Actor
from utils.ds_utils import (
    get_train_ds_config, 
    get_eval_ds_config, 
    get_optimizer_grouped_parameters,
    _z3_params_to_fetch
)
from utils.data.data_utils import create_dataset, DataCollatorReward
from utils.gpu_utils import get_cpu_info, get_gpu_info, get_memory_info

class Strategy(ABC):
    def __init__(   
        self,
        seed : int = 42,
        max_norm : float = 0.0,
        micro_batch_size : int = 1,
        global_batch_size : int = 1,
        zero_stage : int = 2,
        bf16 : bool = True,
        args = None
    ):
        self.args = args
        self.stage = zero_stage
        self.train_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.bf16 = bf16
        self.seed = seed
        self.max_norm = max_norm
        self.adam_offload = getattr(args, 'offload', False)
    
    def set_seed(self, seed : int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def setup_distributed(self, timeout=timedelta(minutes=30)) -> None:
        self.set_seed(self.seed)

        if self.args.local_rank == -1 and "LOCAL_RANK" in os.environ:  # for slurm
            self.args.local_rank = int(os.environ["LOCAL_RANK"])

        if self.args.local_rank != -1:
            torch.cuda.set_device(self.args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed(timeout=timeout)
        self.world_size = dist.get_world_size()
        self.accumulated_gradient = (
            self.train_batch_size // self.micro_batch_size // self.world_size
        )
        self.print(f'Accumulated Steps {self.accumulated_gradient}')
    
    def setup_dataloader(self, tokenizer : PreTrainedTokenizer, dataset_type : str, **kwargs):
        train_dataset, eval_dataset = create_dataset(
            self.args.local_rank,
            self.args.data_path,
            self.args.data_split,
            self.args.data_output_path,
            self.args.data_split_index,
            self.args.seed,
            tokenizer,
            self.args.max_seq_len,
            end_of_conversation_token = tokenizer.eos_token,
            reload = False,
            dataset_type = dataset_type,
            **kwargs
        )
        if dataset_type.lower() in ['reward', 'sentence-reward', 'token-reward']:
            data_collator = DataCollatorReward()
        elif dataset_type.lower() == 'rlhf':
            raise NotImplementedError('You should implemented collator for RLHF firstly!')
        elif dataset_type.lower() == 'sft':
            data_collator = default_data_collator
        else:
            raise NotImplementedError
        
        if self.args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
            eval_sampler = SequentialSampler(eval_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
            eval_sampler = DistributedSampler(eval_dataset)

        train_dataloader = DataLoader(
            train_dataset,
            collate_fn = data_collator,
            sampler = train_sampler,
            batch_size = self.micro_batch_size
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn = data_collator,
            sampler = eval_sampler,
            batch_size = self.micro_batch_size,
        )

        return train_dataloader, eval_dataloader
    
    def create_optimizer(self, model, **kwargs) -> Optimizer:
        # assert isinstance(model, nn.Module)
        AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            model, kwargs['weight_decay']
        )
        optimizer = AdamOptimizer(
            optimizer_grouped_parameters, 
            lr = self.args.learning_rate, 
            betas = (0.9, 0.95)
        )

        return optimizer
    
    def get_ds_train_config(self, tb_name : str = ''):
        ds_config = get_train_ds_config(
            offload = self.adam_offload,
            stage = self.args.zero_stage,
            enable_tensorboard = self.args.enable_tensorboard,
            bf16 = self.args.bf16,
            tb_path = self.args.tensorboard_path,
            tb_name = tb_name
        )
        ds_config['train_micro_batch_size_per_gpu'] = self.micro_batch_size
        ds_config['train_batch_size'] = self.train_batch_size

        return ds_config
    
    def get_ds_eval_config(self, offload : bool = False, zero_stage : int = None):
        zero_stage = 3 if (zero_stage == None or zero_stage == 3) else 0
        ds_config = get_eval_ds_config(
            offload = offload, 
            stage = zero_stage,
            bf16 = self.bf16
        )
        ds_config['train_micro_batch_size_per_gpu'] = self.micro_batch_size
        ds_config['train_batch_size'] = self.train_batch_size

        return ds_config
    
    def ds_init_train_model(self, model, optim, scheduler, ds_config):
        engine, optim, _, scheduler = deepspeed.initialize(
            model = model,
            optimizer = optim,
            lr_scheduler = scheduler,
            config = ds_config,
            args = {'local_rank' : self.args.local_rank},
            dist_init_required = True
        )

        return engine, optim, scheduler
    
    def ds_init_eval_model(self, model, ds_config):
        engine, *_ = deepspeed.initialize(
            model = model,
            args = {'local_rank' : self.args.local_rank},
            config = ds_config,
            dist_init_required = True
        )

        return engine
    
    def is_rank_0(self):
        return dist.get_rank() == 0
    
    def print(self, *msg):
        if self.is_rank_0():
            print(*msg)
    
    def _unwrap_model(self, model : nn.Module) -> nn.Module:
        if hasattr(model, 'module'):
            return model.module
        # elif isinstance(model, Actor):
        #     return model.base_model
        else:
            return model
    
    def save_model(self, model : nn.Module, tokenizer : PreTrainedTokenizer, **kwargs):
        output_dir = self.args.output_dir
        assert output_dir is not None
        if self.is_rank_0():
            os.makedirs(output_dir, exist_ok = True)
        
        unwraped_model = self._unwrap_model(model)
        output_state_dict = {}
        for k, v in unwraped_model.named_parameters():
            params_to_fetch = _z3_params_to_fetch([v])
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled = len(params_to_fetch) > 0):
                vv = v.data.cpu()
                if self.is_rank_0():
                    output_state_dict[k] = vv
        
        if self.is_rank_0():
            state_dict = unwraped_model.state_dict()
            for k, v in unwraped_model.named_buffers():
                if k not in state_dict:
                    continue
                vv = v.data.cpu()
                output_state_dict[k] = vv

            state_dict_keys = set(state_dict.keys())
            output_state_dict_keys = set(output_state_dict.keys())

            if getattr(unwraped_model.config, 'tie_word_embeddings', False) and \
            ('lm_head.weight' in state_dict_keys):
                state_dict_keys.remove('lm_head.weight')
            
            assert state_dict_keys.issubset(
                output_state_dict_keys
            ), f"mismatch keys {output_state_dict_keys.symmetric_difference(state_dict_keys)}"

            if isinstance(unwraped_model, PeftModel):
                unwraped_model.save_pretrained(output_dir, **kwargs)
                if self.stage == 3:
                    torch.save(
                        get_peft_model_state_dict(unwraped_model, output_state_dict),
                        os.path.join(output_dir, 'adapter_model.bin')
                    )
            else:
                # unwraped_model.save_pretrained(output_dir, state_dict=output_state_dict, **kwargs)
                torch.save(output_state_dict, os.path.join(output_dir, 'pytorch_model.bin'))
            
            # save config
            output_config_file = os.path.join(output_dir, "config.json")
            unwraped_model.config.to_json_file(output_config_file)
            # save tokenizer
            tokenizer.save_pretrained(output_dir)
    
    def load_model(
        self,
        model: nn.Module,
        path: str,
        map_location : str = "cpu",
        strict: bool = False,
        key_replace_fn=None,
    ) -> None:
        unwrapped_model = self._unwrap_model(model)
        assert os.path.exists(path)
        supported_file_types = ['.safetensors', '.bin', '.pt', '.pth']
        def load_state_dict(file_path : str, file_type : str):
            assert file_type in supported_file_types, f'Unsupported file type {file_type} to load'
            if file_type == '.safetensors':
                from safetensors import safe_open
                state_dict = {}
                with safe_open(file, framework = 'pt', device = map_location) as f:
                    for k in f.keys():
                        state_dict[k] = f.get_tensor(k)
            elif file_type in ['.bin', '.pt', '.pth']:
                state_dict = torch.load(file_path, map_location = map_location)
            else:
                state_dict = None
            return state_dict
        
        if os.path.isdir(path):
            model_files = []
            file_types = []
            for file in os.listdir(path):
                for file_type in supported_file_types:
                    if file_type in file:
                        model_files.append(file)
                        file_types.append(file_type)
            assert len(model_files) == 1, f'Find too much file {model_files} in {path} to load !'
            path = os.path.join(path, model_files[0])
            file_type = file_types[0]
        else:
            file_type = None
            for supported_file_type in supported_file_types:
                if supported_file_type in path:
                    file_type = supported_file_type
                    break
        state_dict = load_state_dict(path, file_type)
        # state_dict = torch.load(path, map_location = map_location)
        if key_replace_fn:
            state_dict = key_replace_fn(state_dict)
        unwrapped_model.load_state_dict(state_dict, strict=strict)
    
    def all_reduce(self, data, op : str = "mean"):
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            if op == "mean":
                data /= self.world_size
            dist.all_reduce(data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data
        

    def to_device(self, data, device : str = None):
        device = device if device is not None else torch.cuda.current_device()
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = v.to(device)
        else:
            ret = data.to(device)
        
        return ret
    
    def print_machine_info(self):
        cpu_info = get_cpu_info()
        gpu_info = get_gpu_info()
        mem_info = get_memory_info()

        self.print(f"CPU Information:")
        self.print(f'Physical cores: {cpu_info["physical_cores"]}')
        self.print(f'Logical cores: {cpu_info["logical_cores"]}')
        self.print(f'CPU Usage: {cpu_info["usage"]}%')
        self.print(f'Total Memory: {mem_info["total_memory"]}')
        self.print(f'Used Memory: {mem_info["used_memory"]}')

        self.print("\nGPU Information:")
        for idx, info in enumerate(gpu_info):
            self.print(f"GPU {idx}:")
            for key, value in info.items():
                self.print(f"{key}: {value}")
            self.print()
