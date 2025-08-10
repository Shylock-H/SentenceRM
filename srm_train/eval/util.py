from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)
from peft import get_peft_model, LoraConfig
import torch
import math, os, json
from typing import List, Union

from utils.data.spliter import Spliter
from model import SentenceRewardModel, RewardModel

from utils.global_utils import ENS_TOKEN, SAT_MODEL

def create_spliter(
    tokenizer : PreTrainedTokenizer, 
    model_name : str = 'sat-3l', 
    end_sentence_token : str = '<END>',
    end_text_token : str = '</s>',
    device : str = 'cpu'
):
    spliter =  Spliter(
        tokenizer, 
        model_name, 
        end_sentence_token, 
        end_text_token
    )
    if device.lower() != 'cpu' and torch.cuda.is_available():
        spliter.to(torch.cuda.current_device())
    
    return spliter

# load tokenizer
def load_tokenizer(save_path : str, padding_side : str = 'right'):
    tokenizer = AutoTokenizer.from_pretrained(save_path, fast_tokenizer = True)
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer

def load_rm_model(save_path : str, tokenizer : PreTrainedTokenizer):
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
    if 'causal_sentence_mask' in configs.keys():
        kwargs['r_embed_dim'] = configs['reward_embed_dim']
        kwargs['causal_sentence_mask'] = configs['causal_sentence_mask']
        kwargs['weighted_sum'] = configs['weighted_sum']
        rm_type = SentenceRewardModel
    else:
        rm_type = RewardModel
        # rm_type=TokenRewardModel

    rm_model = rm_type(**kwargs)
    
    state_dict = torch.load(os.path.join(save_path, 'pytorch_model.bin'), 'cpu')
    rm_model.load_state_dict(state_dict)
    
    return rm_model

def predict_reward(
    prompts : List[str], 
    responses : List[str], 
    model : Union[RewardModel, SentenceRewardModel], 
    tokenizer : PreTrainedTokenizer,
    max_seq_len : int = 1024,
    spliter : Spliter = None,
    model_type : str = 'rm'
):
    template = ['\n\nHuman: ', '\n\nAssistant:']
    inputs = []
    for prompt, response in zip(prompts, responses):
        if model_type.lower() == 'rm':
            seq = prompt.join(template) + response.rstrip('\n')
            if not seq.endswith(tokenizer.eos_token):
                seq += (' ' + tokenizer.eos_token)
            input_tokens = tokenizer(
                seq, 
                max_length = max_seq_len,
                padding = 'max_length', 
                truncation = True,
                return_tensors = 'pt'
            )
            inputs.append(input_tokens)
        elif model_type.lower() == 'srm':
            assert spliter is not None
            x = prompt.join(template)
            y = spliter.split_texts(' ' + response)[0]
            seq = x + spliter._end_of_sentence_token + y
            input_tokens = spliter.tokenization(seq, max_seq_len)
            inputs.append(input_tokens)
        else:
            raise NotImplementedError
    
    ks = list(inputs[-1].keys())
    model_inputs = {}
    device = model.base_model.device
    for k in ks:
        model_inputs[k] = torch.stack([res[k] for res in inputs]).to(device).view(len(inputs), max_seq_len)
    # predict
    res = model.forward_value(**model_inputs)
    if model_type.lower() == 'rm':
        batch_rewards = res['rewards']
    elif model_type.lower() == 'srm':
        batch_rewards = res['scores']
        # batch_rewards = res['rewards']
    else:
        raise NotImplementedError

    return batch_rewards

class EvalFn:
    def __init__(
        self,
        model : Union[RewardModel, SentenceRewardModel],
        tokenizer : PreTrainedTokenizer,
        max_seq_length : int = 1024,
        model_type : str = 'rm',
        **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.spliter = kwargs['spliter'] if 'spliter' in kwargs.keys() else None
        self.model_type = model_type
        assert self.spliter != None or model_type.lower() == 'rm'
    
    def __call__(
        self, 
        prompts : List[str],
        responses : List[str],
        **kwargs,
    ):
        return predict_reward(
            prompts, responses, self.model, self.tokenizer, self.max_seq_length, self.spliter, self.model_type
        )