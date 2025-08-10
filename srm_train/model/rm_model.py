import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict

# class RewardModel(nn.Module):
#     def __init__(
#         self, 
#         base_model : PreTrainedModel, 
#         tokenizer : PreTrainedTokenizer
#     ):
#         super().__init__()
#         self.config = base_model.config
#         self.hidden_size = None
#         if hasattr(self.config, 'word_embed_proj_dim'):
#             self.hidden_size = self.config.word_embed_proj_dim
#         elif hasattr(self.config, 'hidden_size'):
#             self.hidden_size = self.config.hidden_size
#         else:
#             self.hidden_size = self.config.n_embed
        
#         self.base_model = base_model
#         self.value_head = nn.Linear(self.hidden_size, 1, bias = False)

#         self.tokenizer = tokenizer
    
#     def gradient_checkpointing_enable(self):
#         self.base_model.gradient_checkpointing_enable()
    
#     def gradient_checkpointing_disable(self):
#         self.base_model.gradient_checkpointing_disable()
    
#     def enable_input_require_grads(self):
#         self.base_model.enable_input_require_grads()
    
#     def forward(
#         self,
#         input_ids : torch.Tensor,
#         attention_mask : torch.Tensor,
#         past_key_values : torch.Tensor = None,
#         position_ids : torch.Tensor = None,
#         head_mask : torch.Tensor = None,
#         inputs_embeds : torch.Tensor = None,
#         use_cache : bool = False,
#     ) -> Dict:
        
#         # Train RM
        
#         hidden_states = self.base_model(
#             input_ids,
#             attention_mask = attention_mask,
#             past_key_values = past_key_values,
#             inputs_embeds = inputs_embeds,
#             use_cache = use_cache
#         )[0]

#         values = self.value_head(hidden_states).squeeze(-1)

#         assert len(input_ids.shape) == 2
#         bs, seq_len = input_ids.shape
#         bs = bs // 2

#         eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim = 1, keepdim = True)
#         rewards = values.gather(dim = 1, index = eos_indices).squeeze(-1)
#         chosen_rewards, reject_rewards = rewards[ : bs], rewards[bs : ]
#         loss = -torch.nn.functional.logsigmoid(chosen_rewards - reject_rewards).mean()

#         return dict(
#             loss = loss,
#             chosen_rewards = chosen_rewards,
#             reject_rewards = reject_rewards
#         )
    
#     @torch.no_grad()
#     def forward_value(
#         self,
#         input_ids : torch.Tensor,
#         attention_mask : torch.Tensor,
#         past_key_values : torch.Tensor = None,
#         position_ids : torch.Tensor = None,
#         head_mask : torch.Tensor = None,
#         inputs_embeds : torch.Tensor = None,
#         use_cache : bool = False,
#     ) -> Dict:
#         # evaluate RM
#         hidden_states = self.base_model(
#             input_ids,
#             attention_mask = attention_mask,
#             past_key_values = past_key_values,
#             inputs_embeds = inputs_embeds,
#             use_cache = use_cache
#         )[0]

#         values = self.value_head(hidden_states).squeeze(-1)
#         eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim = 1, keepdim = True)
#         rewards = values.gather(dim = 1, index = eos_indices).squeeze(-1)

#         return dict(
#             rewards = rewards,
#             values = values
#         )
class RewardModel(nn.Module):
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
        # self.value_head = nn.Linear(self.hidden_size, 1, bias = False)
        self.out_head = nn.Linear(self.hidden_size, 2, bias = False)

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

        # values = self.value_head(hidden_states).squeeze(-1)
        logits = self.out_head(hidden_states)
        # shape (bs, seq_len, 2)

        assert len(input_ids.shape) == 2
        bs, seq_len = input_ids.shape
        bs = bs // 2

        eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim = 1, keepdim = True)
        # rewards = values.gather(dim = 1, index = eos_indices).squeeze(-1)
        # chosen_rewards, reject_rewards = rewards[ : bs], rewards[bs : ]
        # loss = -torch.nn.functional.logsigmoid(chosen_rewards - reject_rewards).mean()
        logits = logits[torch.arange(logits.shape[0]), eos_indices.reshape((-1, ))]
        labels = torch.cat([torch.zeros(bs), torch.ones(bs)]).to(input_ids.device, dtype = torch.long)
        loss = nn.CrossEntropyLoss()(logits, labels)
        probs = torch.nn.functional.softmax(logits, dim = 1)
        rewards = torch.log(probs[ : , 0]) - torch.log(probs[ : , 1])
        chosen_rewards = rewards[ : bs]
        reject_rewards = rewards[bs : ]
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
    ) -> Dict:
        # evaluate RM
        hidden_states = self.base_model(
            input_ids,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache
        )[0]

        # values = self.value_head(hidden_states).squeeze(-1)
        # eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim = 1, keepdim = True)
        # rewards = values.gather(dim = 1, index = eos_indices).squeeze(-1)
        logits = self.out_head(hidden_states)
        eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim = 1, keepdim = True)
        logits = logits[torch.arange(logits.shape[0]), eos_indices.reshape((-1, ))]
        probs = torch.nn.functional.softmax(logits, dim = 1)
        rewards = torch.log(probs[ : , 0]) - torch.log(probs[ : , 1])
        values = rewards
        
        return dict(
            rewards = rewards,
            values = values
        )