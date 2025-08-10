
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



class TokenRewardModel(nn.Module):
    def __init__(
        self, 
        base_model : PreTrainedModel, 
        tokenizer : PreTrainedTokenizer,
        r_embed_dim : int = 128,
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
        
        self.tokenizer = tokenizer
        self.r_embed_dim = r_embed_dim
        self.penalty_coef = 0.0

        self.r_num_head = 1
        self.base_model = base_model
        self.r_proj = nn.Linear(self.hidden_size, 1)
        
        self.r_out_fn = nn.Identity()
        
    def gradient_checkpointing_enable(self):
        self.base_model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()
    
    def enable_input_require_grads(self):
        self.base_model.enable_input_require_grads()
    
    def _get_rewards(self, hidden_states : torch.Tensor, split_masks : torch.Tensor):
        v : torch.Tensor = self.r_proj(hidden_states)
        v = self.r_out_fn(v)

        bs, seq_len, _ = v.shape
        
        sentence_rewards = []
        sequence_rewards = []
        
        for i in range(bs):
            split_tokens = torch.where(split_masks[i] == 1)[0]
            if split_tokens.shape[0] == 1: # truncate the last token
                sentence_reward = v[i, split_tokens]
                seq_reward = v[i, split_tokens]
            else:
                sentence_reward = v[i, split_tokens[0] + 1 : split_tokens[-1] + 1]
                seq_reward = v[i, split_tokens[0] + 1 : split_tokens[-1] + 1].mean(dim = 0)
            
            sentence_rewards.append(sentence_reward.squeeze(-1))
            sequence_rewards.append(seq_reward.reshape((-1, 1)))
        
        return dict(
            sentence_rewards = sentence_rewards,
            sequence_rewards = torch.stack(sequence_rewards).squeeze(1),
        )

    def _prepare(self, batch : Dict, dtype : torch.dtype, attention_heads : int = 1):
        assert attention_heads == 1, f'Only support one head attention'
        input_ids, attention_mask, split_mask = itemgetter(
            'input_ids', 'attention_mask', 'split_mask'
        )(batch)
        assert len(input_ids.shape) == 2
        masks = None
        
        return split_mask, masks       
        
    def forward(
        self,
        input_ids : torch.Tensor,
        attention_mask : torch.Tensor,
        splitted_mask : torch.Tensor,
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
        split_token_masks, sentence_masks = self._prepare(
            batch = dict(
                input_ids = input_ids,
                attention_mask = attention_mask,
                split_mask = splitted_mask
            ),
            dtype = hidden_states.dtype
        )
        bs = input_ids.shape[0] // 2
        results = self._get_rewards(hidden_states, split_token_masks)
        rewards = itemgetter('sequence_rewards')(results)
        chosen_rewards, reject_rewards = rewards[ : bs], rewards[bs : ]
        
        loss = 0.0
        total_penalty = 0.0
        penalty_ratio = 0.0
        chosen_scores, reject_scores = [], []

        for i in range(bs):
            chosen_score, reject_score = chosen_rewards[i], reject_rewards[i]
            alpha = 1.0

            bt_loss = -torch.nn.functional.logsigmoid(alpha * (chosen_score - reject_score)).mean()
            penalty_term = torch.zeros_like(bt_loss).detach()
            
            total_penalty += penalty_term.item()
            penalty_ratio += (penalty_term.item() / (bt_loss.item() + penalty_term.item() + 1e-6))
            loss += (bt_loss + penalty_term)

            chosen_scores.append(chosen_score)
            reject_scores.append(reject_score)
        
        loss = loss / bs
        total_penalty = total_penalty / bs
        penalty_ratio = penalty_ratio / bs

        return dict(
            loss = loss,
            chosen_rewards = torch.stack(chosen_scores),
            reject_rewards = torch.stack(reject_scores),
            total_penalty = total_penalty,
            penalty_ratio = penalty_ratio,
        )
    
    @torch.no_grad()
    def forward_value(
        self,
        input_ids : torch.Tensor,
        attention_mask : torch.Tensor,
        splitted_mask : torch.Tensor,
        past_key_values : torch.Tensor = None,
        position_ids : torch.Tensor = None,
        head_mask : torch.Tensor = None,
        inputs_embeds : torch.Tensor = None,
        use_cache : bool = False,
    ) -> Dict:
        # evaluate RM
        hidden_states = self.base_model(
            input_ids,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache
        )[0]
        split_token_masks, sentence_masks = self._prepare(
            batch = dict(
                input_ids = input_ids,
                attention_mask = attention_mask,
                split_mask = splitted_mask
            ),
            dtype = hidden_states.dtype
        )
        results = self._get_rewards(hidden_states, split_token_masks)
        rewards, sentence_rewards = itemgetter('sequence_rewards', 'sentence_rewards')(results)

        return dict(
            rewards = rewards,
            scores = rewards,
            sentence_rewards = sentence_rewards
        )
    
def load_trm_model(save_path : str, tokenizer : PreTrainedTokenizer, bf16: bool = False):
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
    rm_type = TokenRewardModel
    print("TokenRewardModel")

    rm_model = rm_type(**kwargs)
    
    state_dict = torch.load(os.path.join(save_path, 'pytorch_model.bin'), 'cpu')
    rm_model.load_state_dict(state_dict)
    if bf16:
        rm_model = rm_model.to(torch.bfloat16)
    return rm_model