import sys, os, argparse, json
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import torch.distributed
from datetime import datetime
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    PreTrainedModel, 
    PreTrainedTokenizer,
    GenerationConfig
)
from typing import Union, List, Callable, Dict
from datasets import load_dataset
from tqdm import tqdm
import torch
import torch.nn as nn

from model import RewardModel, SentenceRewardModel
from eval.util import load_rm_model, load_tokenizer, create_spliter, EvalFn
from utils.util import get_strategy
from utils.global_utils import ENS_TOKEN, SAT_MODEL, LOCAL_DATASET_DIR
import torch.distributed as dist

class BoNSampler:
    def __init__(
        self,
        base_model : PreTrainedModel,
        tokenizer : PreTrainedTokenizer,
        eval_fn : Callable,
        generate_config : GenerationConfig = None,
        num_samples : int = 32,
        num_candinates : int = 1,
    ):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.eval_fn = eval_fn
        self.N = num_samples
        self.num_candinates = num_candinates
        self.generate_config = generate_config
    
    def extract_prompt_response(
        self, 
        model_input : torch.Tensor, 
        model_output : torch.Tensor
    ):
        prompt = self.tokenizer.batch_decode(model_input, skip_special_tokens = True)
        prompt_length = model_input.shape[-1]
        response = self.tokenizer.batch_decode(model_output[ : , prompt_length : ], skip_special_tokens = True)
        if 'meta-llama' in self.base_model.name_or_path.lower():
            prompt = [prompt[i].lstrip('user\n\n') for i in range(self.N)]
            response = [response[i].lstrip('assistant\n\n') for i in range(self.N)]
        
        return prompt, response
    
    def generate(
        self,
        prompts : Union[List[str], torch.Tensor],
        max_seq_length : int = 1024,
        **generate_kwargs,
    ):
        if isinstance(prompts, torch.Tensor) and len(prompts.shape) == 0:
            model_inputs = prompts.unsqueeze(0)
        elif isinstance(prompts, List):
            assert type(prompts[-1]) == str
            # tokenize
            model_inputs = []
            for prompt in prompts:
                res = self.tokenizer(
                    prompt,
                    padding = 'do_not_pad',
                    truncation = True,
                    max_length = max_seq_length,
                    return_tensors = 'pt',
                    add_special_tokens = False
                )
                model_inputs.append(res['input_ids'].squeeze(0))
        
        model_outputs = []
        gen_outputs = []
        device = self.base_model.device
        with torch.no_grad():
            for model_input in model_inputs:
                model_input = model_input.repeat((self.N, 1))
                output = self.base_model.generate(
                    model_input.to(device),
                    generation_config = self.generate_config,
                    **generate_kwargs
                ).squeeze()
                output = self.tokenizer.batch_decode(output, skip_special_tokens = True)
                prompt = self.tokenizer.batch_decode(model_input, skip_special_tokens = True)
                # TODO: Debug 
                prompt_length = model_input.shape[-1]
                response = [res[prompt_length : ] for res in output]
                score : torch.Tensor = self.eval_fn(prompt, response)
                
                model_outputs.append([
                    output[i] for i in score.topk(self.num_candinates).indices
                ])
                gen_outputs.append([
                    prompt, response, score.detach().cpu().numpy().reshape((self.N, ))
                ])
        
        return model_outputs, gen_outputs
    
    def sample(
        self,
        prompts : Union[List[str], torch.Tensor],
        max_seq_length : int = 1024,
        **generate_kwargs,
    ):
        if isinstance(prompts, torch.Tensor) and len(prompts.shape) == 0:
            model_inputs = prompts.unsqueeze(0)
        elif isinstance(prompts, List):
            assert type(prompts[-1]) == str
            # tokenize
            model_inputs = []
            for prompt in prompts:
                res = self.tokenizer(
                    prompt,
                    padding = 'do_not_pad',
                    truncation = True,
                    max_length = max_seq_length,
                    return_tensors = 'pt',
                    add_special_tokens = False
                )
                model_inputs.append(res['input_ids'].squeeze(0))
        
        gen_outputs = []
        device = self.base_model.device
        generator = f"{self.base_model.name_or_path.split('/')[-1]}-BoN-{self.N}"
        with torch.no_grad():
            for model_input in model_inputs:
                model_input = model_input.repeat((self.N, 1))
                output = self.base_model.generate(
                    model_input.to(device),
                    generation_config = self.generate_config,
                    max_length = max_seq_length,
                    **generate_kwargs
                ).reshape((self.N, -1))
                # prompt = self.tokenizer.batch_decode(model_input, skip_special_tokens = True)
                # prompt_length = model_input.shape[-1]
                # response = self.tokenizer.batch_decode(output[ : , prompt_length : ], skip_special_tokens = True)
                prompt, response = self.extract_prompt_response(model_input, output)
                gen_outputs.append(
                    [
                        {
                            'instruction' : prompt[i],
                            'output' : response[i],
                            'generator' : generator
                        }
                        for i in range(self.N)
                    ]
                )
        
        return gen_outputs

def sample(args):
    print('#' * 15 + ' Sampling ' + '#' * 15)
    strategy = get_strategy(args)
    strategy.setup_distributed()
    strategy.set_seed(seed = 1234)
    ds_config = strategy.get_ds_eval_config()
    torch.distributed.barrier()
    max_seq_length = 1024
    # base model
    base_model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    base_model = strategy.ds_init_eval_model(base_model, ds_config)
    tokenizer = load_tokenizer(base_model_name, 'left')

    dataset_name = 'tatsu-lab/alpaca_eval'
    dataset = load_dataset(dataset_name, 'alpaca_eval')['eval']
    
    eval_fn = None
    # llama-3.1-8b-instruct
    # generation_config = GenerationConfig(
    #     temperature = 0.6, top_k = 50, top_p = 0.9, do_sample = True, pad_token_id = tokenizer.eos_token_id
    # )
    # greedy
    generation_config = GenerationConfig(
        do_sample = False, pad_token_id = tokenizer.eos_token_id
    )
    bon_sampler = BoNSampler(base_model, tokenizer, eval_fn, generation_config, num_samples = 1)
    results = []
    for i in tqdm(range(0, len(dataset))):
        prompt = dataset[i]['instruction']
        prompt = tokenizer.apply_chat_template(
            conversation = [
                {
                 'role' : 'user', 
                 'content' : prompt
                }
            ],
            tokenize = False
        )
        gen_output = bon_sampler.sample([prompt], max_seq_length)
        results.extend(gen_output)
    
    save_dir = os.path.join(LOCAL_DATASET_DIR, 'BoN')
    os.makedirs(save_dir, exist_ok = True)
    # file_name = f"{base_model.name_or_path.split('/')[-1]}-N-{bon_sampler.N}-{dataset_name.split('/')[-1]}.json"
    file_name = f"{base_model.name_or_path.split('/')[-1]}-greedy-{dataset_name.split('/')[-1]}.json"
    with open(os.path.join(save_dir, file_name), 'w', encoding = 'utf-8') as f:
        json.dump(results, f, ensure_ascii = False, indent = 4)
    f.close()    

def evaluate(args):
    strategy = get_strategy(args)
    strategy.setup_distributed()
    strategy.set_seed(seed = 1234)
    ds_config = strategy.get_ds_eval_config()
    torch.distributed.barrier()
    max_seq_length = 1024
    world_size = strategy.world_size
    rank = args.local_rank
    # prepare reward model
    rm_tokenizer = load_tokenizer(args.path)
    rm_model = load_rm_model(args.path, rm_tokenizer)
    model_type = None
    if isinstance(rm_model, SentenceRewardModel):
        spliter = create_spliter(
            rm_tokenizer,
            model_name = SAT_MODEL,
            end_sentence_token = ENS_TOKEN,
            end_text_token = rm_tokenizer.eos_token,
            device = 'cuda'
        )
        # spliter = strategy.ds_init_eval_model(spliter, ds_config)
        model_type = 'srm'
    else:
        spliter = None
        model_type = 'rm'
    
    rm_model = strategy.ds_init_eval_model(rm_model, ds_config)
    eval_kwargs = dict(
        model_type = model_type,
        spliter = spliter
    )
    eval_fn = EvalFn(model = rm_model, tokenizer = rm_tokenizer, max_seq_length = max_seq_length, **eval_kwargs)
    # prepare dataset
    dataset_path = 'dataset_files/BoN/Meta-Llama-3-8B-Instruct-N-32-alpaca_eval.json'
    with open(dataset_path, 'r+') as f:
        dataset_list: List[List[Dict]] = json.load(f)
    f.close()

    # Convert to torch Dataset
    from torch.utils.data import Dataset, DataLoader, DistributedSampler
    
    class BonDataset(Dataset):
        def __init__(self, data_list):
            self.data = data_list
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            return self.data[idx]

    dataset = BonDataset(dataset_list)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 可以根据需要调整
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    results = []
    for batch in tqdm(
            dataloader,
            desc=f'[Rank {rank}] Evaluating', 
            position=rank,
            leave=False
        ):
        prompts = [data_dict['instruction'][0] for data_dict in batch]
        responses = [data_dict['output'][0] for data_dict in batch]
        scores = eval_fn(prompts, responses)
        score, candinate_index = torch.topk(scores, 1)
        score, candinate_index = score.item(), candinate_index.item()
        results.append(
            {
                'instruction': prompts[candinate_index],
                'output': responses[candinate_index],
                'score': score    
            }
        )

    if world_size > 1:
        dist.barrier()
        all_results = [None] * world_size
        dist.all_gather_object(all_results, results)
        if rank == 0:
            results = [item for sublist in all_results for item in sublist]

    if rank == 0:
        save_dir = os.path.join(LOCAL_DATASET_DIR, 'BoN')
        os.makedirs(save_dir, exist_ok = True)
        current_time = datetime.now().strftime("%Y.%m.%d-%H.%M")
        file_name = f"result_{model_type}_{current_time}.json"
        with open(os.path.join(save_dir, file_name), 'w', encoding = 'utf-8') as f:
            json.dump(results, f, ensure_ascii = False, indent = 4)
        f.close()

def evaluate_all(args):
    strategy = get_strategy(args)
    strategy.setup_distributed()
    strategy.set_seed(seed = 1234)
    ds_config = strategy.get_ds_eval_config()
    torch.distributed.barrier()
    max_seq_length = 1024
    world_size = strategy.world_size
    rank = args.local_rank
    # prepare reward model
    rm_tokenizer = load_tokenizer(args.path)
    rm_model = load_rm_model(args.path, rm_tokenizer, args.rm_type)
    model_type = None
    if isinstance(rm_model, SentenceRewardModel):
        spliter = create_spliter(
            rm_tokenizer,
            model_name = SAT_MODEL,
            end_sentence_token = ENS_TOKEN,
            end_text_token = rm_tokenizer.eos_token,
            device = 'cuda'
        )
        # spliter = strategy.ds_init_eval_model(spliter, ds_config)
        model_type = 'srm'
    else:
        spliter = None
        model_type = 'rm'
    
    rm_model = strategy.ds_init_eval_model(rm_model, ds_config)
    eval_kwargs = dict(
        model_type = model_type,
        spliter = spliter
    )
    eval_fn = EvalFn(model = rm_model, tokenizer = rm_tokenizer, max_seq_length = max_seq_length, **eval_kwargs)
    # prepare dataset
    dataset_path = 'dataset_files/BoN/Meta-Llama-3-8B-Instruct-N-32-alpaca_eval.json'
    with open(dataset_path, 'r+') as f:
        dataset_list: List[List[Dict]] = json.load(f)
    f.close()

    # Convert to torch Dataset
    from torch.utils.data import Dataset, DataLoader, DistributedSampler
    
    class BonDataset(Dataset):
        def __init__(self, data_list):
            self.data = data_list
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            return self.data[idx]

    dataset = BonDataset(dataset_list)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 可以根据需要调整
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    results = []
    cnt = 1
    for batch in tqdm(
            dataloader,
            desc=f'[Rank {rank}] Evaluating', 
            position=rank,
            leave=False
        ):
        batch_results = []
        # Each batch contains a list of 32 responses for one prompt
        prompts = [data_dict['instruction'][0] for data_dict in batch]
        responses = [data_dict['output'][0] for data_dict in batch]
        scores = eval_fn(prompts, responses)
        
        # Group all responses and scores for this prompt
        batch_results = {
            'instruction': prompts[0],  # All prompts are the same in the batch
            'responses': [
                {
                    'output': responses[i],
                    'score': scores[i].item()
                } for i in range(len(responses))
            ]
        }
        results.append(batch_results)
        if cnt == 2:
            break
        cnt += 1

    if world_size > 1:
        dist.barrier()
        all_results = [None] * world_size
        dist.all_gather_object(all_results, results)
        if rank == 0:
            results = [item for sublist in all_results for item in sublist]

    if rank == 0:
        save_dir = os.path.join(LOCAL_DATASET_DIR, 'BoN')
        os.makedirs(save_dir, exist_ok = True)
        current_time = datetime.now().strftime("%Y.%m.%d-%H.%M")
        file_name = f"result_{model_type}_{current_time}.json"
        with open(os.path.join(save_dir, file_name), 'w', encoding = 'utf-8') as f:
            json.dump(results, f, ensure_ascii = False, indent = 4)
        f.close()
          

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type = str)
    parser.add_argument('--local_rank', type = int, default=-1)
    parser.add_argument('--store_all_scores',
        action="store_true",
        help="Whether to store all response scores",
    )
    parser.add_argument('--rm_type',
        type='str',
        default='rm'
    )
    # usage example:
    # deepspeed --include localhost:6,7 eval/bon.py --path /home/liyc/workspace/zxq/sentencerm/log_debug/sentence_reward_model-meta-llama_Llama-3.1-8B-2025-01-19-20-57-52-1234 > /home/liyc/workspace/zxq/sentencerm/log_debug/sentence_reward_model-meta-llama_Llama-3.1-8B-2025-01-19-20-57-52-1234/bon.log
    args = parser.parse_args()

    # sample(args)
    # evaluate(args)
    evaluate_all(args)