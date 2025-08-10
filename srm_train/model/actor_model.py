# modified from https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/deepspeed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim = -1)
    log_probs_labels = log_probs.gather(dim = -1, index = labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class Actor(nn.Module):
    def __init__(
        self, 
        base_model : PreTrainedModel,
        **kwargs,
    ):
        super().__init__()
        self.base_model = base_model
    
    # def backward(self, loss : torch.Tensor):
    #     return self.base_model.backward(loss)
    
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
        num_actions : int = None,
        return_log_probs : bool = False,
        **kwargs,
    ):
        # position_ids = attention_mask.long().cumsum(-1) - 1
        # position_ids.masked_fill_(attention_mask == 0, 1)
        output = self.base_model(input_ids, attention_mask = attention_mask, **kwargs)
        if return_log_probs:
            log_probs = log_probs_from_logits(output['logits'][ : , : -1, : ], input_ids[ : , 1 : ])
            num_actions = 0 if num_actions is None else num_actions
            output['action_log_probs'] = log_probs[-num_actions : ]
        
        return output
    
    def generate(
        self,
        input_ids : torch.Tensor,
        attention_mask : torch.Tensor,
        **kwargs,
    ):
        generate_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": True,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens", 1),
        }

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        # Call generate
        sequences = self.base_model.generate(**generate_args)

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]
    
        return self.process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id)
    
    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1 : -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask