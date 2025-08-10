
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoConfig, AutoModel
from operator import itemgetter
import os
import json
import math
from peft import LoraConfig, get_peft_model
from operator import itemgetter

class BaselineRewardModel(nn.Module):
    def __init__(
        self, 
        base_model : PreTrainedModel, 
        tokenizer : PreTrainedTokenizer
    ):
        super().__init__()
        self.config = base_model.config
        self.hidden_size = None
        if hasattr(self.config, 'word_embed_proj_dim'):
            self.hidden_size = self.config.word_embed_proj_dim
        elif hasattr(self.config, 'hidden_size'):
            self.hidden_size = self.config.hidden_size
        else:
            self.hidden_size = self.config.n_embed
        
        self.base_model = base_model
        self.value_head = nn.Linear(self.hidden_size, 1, bias = False)

        self.tokenizer = tokenizer
    
    def gradient_checkpointing_enable(self):
        self.base_model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()
    
    def enable_input_require_grads(self):
        self.base_model.enable_input_require_grads()
    
    def forward(
        self,
        input_ids : torch.Tensor,
        attention_mask : torch.Tensor,
        past_key_values : torch.Tensor = None,
        position_ids : torch.Tensor = None,
        head_mask : torch.Tensor = None,
        inputs_embeds : torch.Tensor = None,
        use_cache : bool = False,
    ) -> Dict:
        
        # Train RM
        
        hidden_states = self.base_model(
            input_ids,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache
        )[0]

        values = self.value_head(hidden_states).squeeze(-1)

        assert len(input_ids.shape) == 2
        bs, seq_len = input_ids.shape
        bs = bs // 2

        eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim = 1, keepdim = True)
        rewards = values.gather(dim = 1, index = eos_indices).squeeze(-1)
        chosen_rewards, reject_rewards = rewards[ : bs], rewards[bs : ]
        loss = -torch.nn.functional.logsigmoid(chosen_rewards - reject_rewards).mean()

        return dict(
            loss = loss,
            chosen_rewards = chosen_rewards,
            reject_rewards = reject_rewards
        )
    
    @torch.no_grad()
    def forward_value(
        self,
        input_ids : torch.Tensor,
        attention_mask : torch.Tensor,
        past_key_values : torch.Tensor = None,
        position_ids : torch.Tensor = None,
        head_mask : torch.Tensor = None,
        inputs_embeds : torch.Tensor = None,
        use_cache : bool = False,
    ) -> torch.Tensor:
        hidden_states = self.base_model(
            input_ids,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache
        )[0]

        values = self.value_head(hidden_states).squeeze(-1)
        eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim = 1, keepdim = True)
        rewards = values.gather(dim = 1, index = eos_indices).squeeze(-1)

        return {
            'rewards' : rewards
        }


def load_rm_model(save_path : str, tokenizer : PreTrainedTokenizer, bf16 : bool = False):
    assert os.path.isdir(save_path)
    model_config = AutoConfig.from_pretrained(save_path)
    model_class = AutoModel._model_mapping[type(model_config)]
    model : PreTrainedModel = model_class.from_pretrained(
        save_path, config = model_config
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(
        int(8 * math.ceil(len(tokenizer) / 8.0))
    )
    config_path = os.path.join(save_path, "args.json")
    configs = {}
    with open(config_path, "r+") as f:
        for line in f.readlines():
            x = json.loads(line.strip())
            configs.update({k: v for k, v in x.items()})
    if configs["lora_rank"] > 0:
        lora_config = LoraConfig(
            r=configs["lora_rank"],
            lora_alpha=configs["lora_alpha"],
            target_modules=configs["target_modules"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    
    kwargs = {
        'base_model' : model,
        'tokenizer' : tokenizer
    }
    rm_type = BaselineRewardModel
    print("BaselineRewardModel")

    rm_model = rm_type(**kwargs)
    
    state_dict = torch.load(os.path.join(save_path, 'pytorch_model.bin'), 'cpu')
    rm_model.load_state_dict(state_dict)
    if bf16:
        rm_model = rm_model.to(torch.bfloat16)
    return rm_model
