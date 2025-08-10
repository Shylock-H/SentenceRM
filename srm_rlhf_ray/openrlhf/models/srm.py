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

def rotary_emb(x, freqs_cis):
    """Applies RoPE to the input tensor using complex multiplication."""
    # x: [..., seq_len, dim]
    # freqs_cis: [seq_len, dim // 2] (complex) - should be on the same device as x
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2) # [..., seq_len, dim // 2, 2]
    x_complex = torch.view_as_complex(x_reshaped)      # [..., seq_len, dim // 2]

    # Ensure freqs_cis is on the correct device
    freqs_cis = freqs_cis.to(x_complex.device)

    # Perform complex multiplication (element-wise)
    x_rotated_complex = x_complex * freqs_cis # [..., seq_len, dim // 2]

    # Convert back to real and reshape
    x_rotated_real = torch.view_as_real(x_rotated_complex) # [..., seq_len, dim // 2, 2]
    x_out = x_rotated_real.flatten(-2)                     # [..., seq_len, dim]

    return x_out.type_as(x)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device: torch.device = 'cpu'):
    """Precomputes frequency components for RoPE and returns as complex numbers."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs) # Shape: [end, dim // 2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Shape: [end, dim // 2], complex64
    return freqs_cis

class SentenceRewardModel(nn.Module):
    def __init__(
        self, 
        base_model : PreTrainedModel, 
        tokenizer : PreTrainedTokenizer,
        r_embed_dim : int = 128,
        causal_sentence_mask : bool = False,
        weighted_sum : bool = True,
        penalty_coef : float = 0.0,
        use_rope: bool = True, # Add flag to control RoPE
        rope_theta: float = 10000.0, # RoPE parameter
        max_rope_sentences: int = 200 # Max sentences for precomputation
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
        self._causal_rm_mask = causal_sentence_mask
        self._weighted_sum = weighted_sum
        self.penalty_coef = penalty_coef
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.max_rope_sentences = max_rope_sentences

        # self._causal_rm_mask = True
        self.r_num_head = 1
        self.base_model = base_model
        # Reward attention
        if self._weighted_sum:
            self.q_proj = nn.Linear(self.hidden_size, self.r_embed_dim)
            self.k_proj = nn.Linear(self.hidden_size, self.r_embed_dim)
            nn.init.orthogonal_(self.q_proj.weight)
            nn.init.orthogonal_(self.k_proj.weight)
            self.r_proj = nn.Linear(self.hidden_size, 1)
        else:
            self.r_proj = nn.Linear(self.hidden_size, 1)
        
        self.r_out_fn = nn.Identity()

        # Precompute RoPE frequencies if enabled
        if self.use_rope and self._weighted_sum:
            # Detach helps if base_model is already on GPU but we compute this on CPU first
            freqs_cis = precompute_freqs_cis(
                self.r_embed_dim, self.max_rope_sentences, self.rope_theta
            ).detach()
            # Register as buffer to automatically handle device placement
            self.register_buffer('rope_freqs_cis', freqs_cis, persistent=False)
        else:
            self.register_buffer('rope_freqs_cis', None, persistent=False)
        
    def gradient_checkpointing_enable(self):
        self.base_model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()
    
    def enable_input_require_grads(self):
        self.base_model.enable_input_require_grads()
    
    def _get_rewards(self, hidden_states : torch.Tensor, split_masks : torch.Tensor):
        dtype = hidden_states.dtype
        device = hidden_states.device # Get device from input tensor

        if self._weighted_sum:
            q_all : torch.Tensor = self.q_proj(hidden_states)
            k_all : torch.Tensor = self.k_proj(hidden_states)
        else:
             q_all = torch.zeros_like(hidden_states[..., :self.r_embed_dim])
             k_all = torch.zeros_like(hidden_states[..., :self.r_embed_dim])

        v : torch.Tensor = self.r_proj(hidden_states)
        v = self.r_out_fn(v)

        bs, seq_len, _ = hidden_states.shape

        sentence_rewards = []
        sequence_rewards = []
        sentence_nums = []

        # RoPE frequencies are now precomputed in self.rope_freqs_cis (on the correct device)

        for i in range(bs):
            split_tokens = torch.where(split_masks[i] == 1)[0]
            num_sentences = split_tokens.shape[0]

            if num_sentences == 0:
                 vs = v[i, -1:]
                 seq_reward = vs.sum().unsqueeze(0)
                 sentence_rewards.append(vs.reshape((-1,)))
                 sequence_rewards.append(seq_reward)
                 sentence_nums.append(1) # Or 0 depending on desired behavior
                 continue

            # Ensure we don't exceed precomputed RoPE length
            if self.use_rope and self._weighted_sum and num_sentences > self.max_rope_sentences:
                 raise ValueError(
                    f"Number of sentences ({num_sentences}) exceeds precomputed RoPE length "
                    f"({self.max_rope_sentences}). Increase max_rope_sentences during init."
                 )

            vs = v[i, split_tokens]

            if self._weighted_sum:
                q_split = q_all[i, split_tokens] # (num_sentences, r_embed_dim)
                k_split = k_all[i, split_tokens] # (num_sentences, r_embed_dim)

                if self.use_rope:
                    # Select precomputed frequencies for the current number of sentences
                    # self.rope_freqs_cis should already be on the correct device
                    freqs_cis_i = self.rope_freqs_cis[:num_sentences] # Select first num_sentences freqs
                    q_split = rotary_emb(q_split, freqs_cis_i)
                    k_split = rotary_emb(k_split, freqs_cis_i)

                sequence_query = q_split[-1]
                sentence_keys = k_split

                sentence_reward_diffs = vs[1:] - vs[:-1]
                sentence_reward = torch.cat([vs[:1], sentence_reward_diffs], dim=0)

                assert sentence_reward.shape[0] == num_sentences
                assert sentence_keys.shape[0] == num_sentences

                attn_weights = torch.matmul(sequence_query, sentence_keys.transpose(0, 1)) / (q_split.shape[-1] ** 0.5)

                if dtype == torch.float16:
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
                else:
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

                seq_reward = torch.matmul(attn_weights.unsqueeze(0), sentence_reward).squeeze(0)
                sentence_reward = (sentence_reward * attn_weights.unsqueeze(-1)).reshape(1, -1)

            else:
                sentence_reward = vs
                seq_reward = torch.sum(vs).unsqueeze(0)

            sentence_rewards.append(sentence_reward.reshape((-1,)))
            sequence_rewards.append(seq_reward)
            sentence_nums.append(num_sentences)

        return dict(
            sentence_rewards = sentence_rewards,
            sequence_rewards = torch.stack(sequence_rewards).squeeze(1),
            sentence_nums = sentence_nums
        )

    def _prepare(self, batch : Dict, dtype : torch.dtype, attention_heads : int = 1):
        assert attention_heads == 1, f'Only support one head attention'
        input_ids, attention_mask, split_mask = itemgetter(
            'input_ids', 'attention_mask', 'splitted_mask'
        )(batch)
        assert len(input_ids.shape) == 2
        masks = None
        
        return split_mask, masks
    
    def _update_relative_rms(self, sentence_rewards: List[torch.Tensor], is_weighted: bool = False):
        # Get the appropriate statistics based on whether this is weighted or not
        if is_weighted:
            mu = self._rms_mu_w
            var = self._rms_var_w
        else:
            mu = self._rms_mu
            var = self._rms_var

        for sentence_reward in sentence_rewards:
            # print(sentence_reward.shape)
            assert len(sentence_reward.shape) == 1
            T = sentence_reward.shape[0]
            src_pos = np.arange(T)
            relative_pos = np.arange(1, T + 1) / T
            target_pos = (mu.shape[0] * relative_pos).astype(int)
            relative_sentence_reward = torch.zeros(
                mu.shape, 
                dtype = mu.dtype
            ).to(sentence_reward.device)
            target_pos_cnt = torch.zeros(
                mu.shape,
                dtype = mu.dtype
            ).to(sentence_reward.device)
            
            for src_p, target_p in zip(src_pos, target_pos):
                target_p = target_p - 1 if target_p > 0 else target_p
                target_pos_cnt[target_p] += 1
                relative_sentence_reward[target_p] += (
                    (sentence_reward[src_p] - relative_sentence_reward[target_p]) / target_pos_cnt[target_p]
                )
            
            if not is_weighted:
                self._rms_n += 1
                n = self._rms_n
            else:
                self._rms_mu_w += 1
                n = self._rms_mu_w

            mu_last = mu.clone()
            mu.copy_(mu_last + (relative_sentence_reward - mu_last) / n)
            term_1 = (n - 1) * var
            term_2 = (relative_sentence_reward - mu) * (relative_sentence_reward - mu_last)
            var.copy_((term_1 + term_2) / n)

    def _update_rms(self, sentence_rewards : List[torch.Tensor]):
        for sentence_reward in sentence_rewards:
            # reshape
            assert len(sentence_reward.shape) == 1
            # padding
            n = sentence_reward.shape[0]
            if n < self.max_N:
                sentence_reward = torch.nn.functional.pad(
                    sentence_reward, 
                    (0, self.max_N - n), 
                    mode='constant', 
                    value = 0
                )
            # truncate
            elif n > self.max_N:
                sentence_reward = sentence_reward[ : self.max_N]
            
            # update mu, var
            self._rms_n += 1
            mu_last = self._rms_mu.data
            # self._rms_mu = self._rms_mu + (sentence_reward - self._rms_mu) / self._rms_n
            self._rms_mu.data.copy_(mu_last + (sentence_reward - mu_last) / self._rms_n)
            term_1 = (self._rms_n - 1) * self._rms_var.data
            term_2 = (sentence_reward - self._rms_mu) * (sentence_reward - mu_last)
            self._rms_var.data.copy_((term_1 + term_2) / self._rms_n)
                
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
                splitted_mask = splitted_mask
            ),
            dtype = hidden_states.dtype
        )
        bs = input_ids.shape[0] // 2
        results = self._get_rewards(hidden_states, split_token_masks)
        rewards, sentence_rewards, sentence_nums = itemgetter('sequence_rewards', 'sentence_rewards', 'sentence_nums')(results)
        chosen_rewards, reject_rewards = rewards[ : bs], rewards[bs : ]
        chosen_sentence_nums, reject_sentence_nums = sentence_nums[ : bs], sentence_nums[bs : ]
        
        loss = 0.0
        total_penalty = 0.0
        penalty_ratio = 0.0
        max_sentence_nums = 0
        chosen_scores, reject_scores = [], []

        for i in range(bs):
            chosen_score, reject_score = chosen_rewards[i], reject_rewards[i]
            chosen_sentence_num, reject_sentence_num = chosen_sentence_nums[i], reject_sentence_nums[i]
            alpha = 1.0

            bt_loss = -torch.nn.functional.logsigmoid(alpha * (chosen_score - reject_score)).mean()
            if self.penalty_coef > 0.0:
                penalty_term = (chosen_score ** 2 + reject_score ** 2) * self.penalty_coef
            else:
                penalty_term = torch.zeros_like(bt_loss).detach()
            
            total_penalty += penalty_term.item()
            penalty_ratio += (penalty_term.item() / (bt_loss.item() + penalty_term.item() + 1e-6))
            loss += (bt_loss + penalty_term)

            chosen_scores.append(chosen_score)
            reject_scores.append(reject_score)
            max_sentence_nums = max(max_sentence_nums, chosen_sentence_num, reject_sentence_num)
        
        loss = loss / bs
        total_penalty = total_penalty / bs
        penalty_ratio = penalty_ratio / bs

        return dict(
            loss = loss,
            chosen_rewards = torch.stack(chosen_scores),
            reject_rewards = torch.stack(reject_scores),
            total_penalty = total_penalty,
            penalty_ratio = penalty_ratio,
            N = max_sentence_nums
        )
    
    @torch.no_grad()
    def forward_value(
        self,
        input_ids : torch.Tensor,
        attention_mask : torch.Tensor,
        splitted_mask : torch.Tensor, # Ensure this is passed correctly
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

        # Prepare the masks - ensure 'splitted_mask' is passed in the dict
        split_token_masks, _ = self._prepare(
            batch = dict(
                input_ids = input_ids,
                attention_mask = attention_mask,
                splitted_mask = splitted_mask # Pass the input splitted_mask here
            ),
            dtype = hidden_states.dtype
        )

        # Get rewards using the prepared masks
        results = self._get_rewards(hidden_states, split_token_masks)
        rewards, sentence_rewards = itemgetter('sequence_rewards', 'sentence_rewards')(results)

        return dict(
            rewards = rewards,
            scores = rewards,
            sentence_rewards = sentence_rewards
        )


def load_srm_model(save_path : str, tokenizer : PreTrainedTokenizer, bf16: bool = False):
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
    kwargs['r_embed_dim'] = configs['reward_embed_dim']
    kwargs['causal_sentence_mask'] = configs['causal_sentence_mask']
    kwargs['weighted_sum'] = configs['weighted_sum']
    rm_type = SentenceRewardModel
    print("SentenceRewardModel")

    rm_model = rm_type(**kwargs)
    
    state_dict = torch.load(os.path.join(save_path, 'pytorch_model.bin'), 'cpu')
    rm_model.load_state_dict(state_dict)
    if bf16:
        rm_model = rm_model.to(torch.bfloat16)
    return rm_model