from utils.strategy import Strategy
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForCausalLM, 
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from model import RewardModel, SentenceRewardModel, Actor
from peft import get_peft_model, LoraConfig
import math, os
from typing import Union, List, Dict

def get_tokenizer(model_name_or_path : str, padding_side : str = 'right', fast_tokenizer : bool = True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, fast_tokenizer = fast_tokenizer
    )
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer

def get_strategy(args):
    strategy = Strategy(
        seed = getattr(args, 'seed', 42),
        max_norm = getattr(args, 'max_norm', 1.0),
        micro_batch_size = getattr(args, 'per_device_train_batch_size', 1),
        global_batch_size = getattr(args, 'train_batch_size', 32),
        zero_stage = getattr(args, 'zero_stage', 2),
        bf16 = getattr(args, 'bf16', True),
        args = args
    )

    return strategy

def get_base_model(model_name_or_path : str, strategy : Strategy, tokenizer : PreTrainedTokenizer, use_embed_out : bool = False):
    model_config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code = True
    )
    if getattr(strategy.args, 'disable_dropout', False):
        model_config.dropout = 0.0
    try:
        model_type = AutoModelForCausalLM if use_embed_out else AutoModel
        model_class = model_type._model_mapping[type(model_config)]
        model : PreTrainedModel = model_class.from_pretrained(
            model_name_or_path,
            config = model_config
        )
    except Exception as e:
        print("Failed to load from AutoModel!")
        return
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(
        int(8 * math.ceil(len(tokenizer) / 8.0))
    )
    if getattr(strategy.args, 'lora_rank', 0) > 0:
        strategy.print('> Enable LoRA Finetuning')
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r = getattr(strategy.args, 'lora_rank', 8),
            lora_alpha = getattr(strategy.args, 'lora_alpha', 8),
            target_modules = getattr(strategy.args, 'target_modules', None),
            lora_dropout = getattr(strategy.args, 'lora_dropout', 0.0),
            bias = "none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model

def get_actor_model(model_name_or_path : str, strategy : Strategy, tokenizer : PreTrainedTokenizer):
    import time
    start = time.time()
    base_model = get_base_model(
        model_name_or_path, strategy, tokenizer, True
    )
    actor = Actor(base_model)
    end = time.time()
    strategy.print(f'> Creating model from config took {end - start} seconds')
    if os.path.exists(model_name_or_path):
        start = time.time()
        strategy.load_model(actor, model_name_or_path, 'cpu')
        end = time.time()
        strategy.print(f'> Loading state dict from {model_name_or_path} took {end - start} seconds')
    
    return actor

def get_reward_model(model_name_or_path : str, strategy : Strategy, tokenizer : PreTrainedTokenizer):
    import time
    start = time.time()
    base_model = get_base_model(
        model_name_or_path, strategy, tokenizer, False
    )
    rm_model = RewardModel(base_model, tokenizer)
    end = time.time()
    strategy.print(f'> Creating model from config took {end - start} seconds')
    if os.path.exists(model_name_or_path):
        start = time.time()
        strategy.load_model(rm_model, model_name_or_path, 'cpu')
        end = time.time()
        strategy.print(f'> Loading state dict from {model_name_or_path} took {end - start} seconds')
    
    return rm_model

def get_sentence_reward_model(
    model_name_or_path : str, 
    strategy : Strategy, 
    tokenizer : PreTrainedTokenizer,
    reward_embed_dim : int = 128,
    causal_sentence_mask : bool = False,
    weighted_sum : bool = True,
    penalty_coef : float = 0.0
):
    import time
    start = time.time()
    base_model = get_base_model(
        model_name_or_path, strategy, tokenizer, False
    )
    rm_model = SentenceRewardModel(
        base_model, 
        tokenizer, 
        r_embed_dim = reward_embed_dim, 
        causal_sentence_mask = causal_sentence_mask,
        penalty_coef = penalty_coef,
        weighted_sum = weighted_sum
    )
    end = time.time()
    strategy.print(f'> Creating model from config took {end - start} seconds')
    if os.path.exists(model_name_or_path):
        start = time.time()
        strategy.load_model(rm_model, model_name_or_path, 'cpu')
        end = time.time()
        strategy.print(f'> Loading state dict from {model_name_or_path} took {end - start} seconds')
    
    return rm_model


def add_tokens(tokenizer : PreTrainedTokenizer, new_tokens : Union[str, List[str]]):
    new_tokens = [new_tokens] if type(new_tokens) == str else new_tokens
    tokenizer.add_tokens(new_tokens, special_tokens = True)
    print('#' * 30)
    for new_token in new_tokens:
        print(f'Add {new_token} | ID {tokenizer.encode(new_token)[-1]}')
    
    return tokenizer

def convert_token_to_id(token : str, tokenizer : PreTrainedTokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")