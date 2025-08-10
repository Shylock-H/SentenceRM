import sys, os, argparse

import torch.distributed
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import List, Dict, Union
import torch

from utils.data.spliter import Spliter
from model import RewardModel, SentenceRewardModel
from utils.util import get_strategy
from utils.global_utils import ENS_TOKEN, SAT_MODEL
from eval.util import load_rm_model, load_tokenizer, create_spliter


REWARD_BENCH_TO_CATEGORY_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}

def load_reward_bench_dataset():
    data = load_dataset("allenai/reward-bench")["filtered"]
    eval_data = []
    eval_metadata = []
    for example in data:
        eval_data.append({
            "id": f"{example['id']}-chosen",
            "prompt": example["prompt"],
            "response": example["chosen"]
        })
        eval_data.append({
            "id": f"{example['id']}-rejected",
            "prompt": example["prompt"],
            "response": example["rejected"]
        })
        eval_metadata.append({
            "id": str(example["id"]),
            "subset": example["subset"]
        })
    return eval_data, eval_metadata

def predict_reward(
    prompts : List[str], 
    responses : List[str], 
    model : PreTrainedModel, 
    tokenizer : PreTrainedTokenizer,
    max_seq_len : int = 1024,
    spliter : Spliter = None,
    model_type : str = 'rm'
):
    template = ['\n\nHuman: ', '\n\nAssistant: ']
    inputs = []
    for prompt, response in zip(prompts, responses):
        if model_type.lower() == 'rm':
            seq = prompt.join(template) + ' ' + response.rstrip('\n')
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
            y = spliter.split_texts(response)[0]
            seq = x + spliter._end_of_sentence_token + y
            input_tokens = spliter.tokenization(seq, max_seq_len)
            inputs.append(input_tokens)
        elif model_type.lower() == 'trm':
            assert spliter is not None
            x = prompt.join(template)
            y = spliter.split_texts(response)[0]
            seq = x + spliter._end_of_sentence_token + y
            input_tokens = spliter.tokenization(seq, max_seq_len)
            inputs.append(input_tokens)
        else:
            raise NotImplementedError
    
    ks = list(inputs[-1].keys())
    model_inputs = {}
    for k in ks:
        model_inputs[k] = torch.stack([res[k] for res in inputs]).to(torch.cuda.current_device()).view(len(inputs), max_seq_len)
    # predict
    res = model.forward_value(**model_inputs)
    if model_type.lower() == 'rm':
        batch_rewards = res['rewards']
    elif model_type.lower() == 'srm':
        batch_rewards = res['scores']
    elif model_type.lower() == 'trm':
        batch_rewards = res['rewards']
    else:
        raise NotImplementedError

    return batch_rewards

def generate_rewards(
    model : Union[RewardModel, SentenceRewardModel], 
    tokenizer : PreTrainedTokenizer, 
    eval_data : List, 
    batch_size : int,
    max_seq_len : int = 1024,
    spliter : Spliter = None,
    model_type : str = 'rm'
):
    rewards = {}

    for i in tqdm(range(0, len(eval_data), batch_size)):
        batch = eval_data[i : i + batch_size]
        
        prompts = [item["prompt"] for item in batch]
        responses = [item["response"] for item in batch]
        ids = [item["id"] for item in batch]

        batch_rewards = predict_reward(prompts, responses, model, tokenizer, max_seq_len, spliter, model_type)
        
        for id_, reward in zip(ids, batch_rewards):
            rewards[id_] = reward
    
    torch.distributed.barrier()

    return rewards

def post_process_reward_bench(eval_metadata, rewards):
    per_category_scores = {category: [] for category in REWARD_BENCH_TO_CATEGORY_MAPPING.keys()}
    for example in eval_metadata:
        id_ = example["id"]
        chosen_reward = rewards[id_ + "-chosen"]
        rejected_reward = rewards[id_ + "-rejected"]
        for category, subsets in REWARD_BENCH_TO_CATEGORY_MAPPING.items():
            if example["subset"] in subsets:
                per_category_scores[category].append(int(chosen_reward > rejected_reward))
                break
    per_category_scores = {category: np.mean(scores) * 100 for category, scores in per_category_scores.items()}
    per_category_scores["Average"] = np.mean([score for score in per_category_scores.values()])

    # Print scores in a pretty way
    print("\nReward Bench Scores:")
    print("=" * 40)
    max_category_length = max(len(category) for category in per_category_scores.keys())
    for category, score in per_category_scores.items():
        print(f"{category:<{max_category_length}} : {score:.2f}%")
    print("=" * 40)

    return per_category_scores

def main(args):
    strategy = get_strategy(args)
    strategy.setup_distributed()
    ds_config = strategy.get_ds_eval_config()
    torch.distributed.barrier()
    tokenizer = load_tokenizer(args.path)
    model = load_rm_model(args.path, tokenizer)
    
    if isinstance(model, SentenceRewardModel):
        model_type = 'srm'
    elif isinstance(model, RewardModel):
        model_type = 'rm'
    else:
        raise NotImplementedError

    if model_type == 'srm':
        spliter = create_spliter(
            tokenizer, 
            model_name = SAT_MODEL, 
            end_sentence_token = ENS_TOKEN, 
            end_text_token = tokenizer.eos_token
        )
        spliter.to(torch.cuda.current_device())
    else:
        spliter = None

    model = strategy.ds_init_eval_model(model, ds_config)
    # prepare dataset
    ds, sub_ds = load_reward_bench_dataset()
    rewards = generate_rewards(model, tokenizer, ds, 32, model_type = model_type, max_seq_len = 1024, spliter = spliter)
    results = post_process_reward_bench(sub_ds, rewards)
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type = str)
    parser.add_argument('--local_rank', type = int, default=-1)
    args = parser.parse_args()

    main(args)