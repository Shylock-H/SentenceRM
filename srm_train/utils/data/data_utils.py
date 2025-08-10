import warnings
import torch
import torch.distributed
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import os
import hashlib
from copy import deepcopy
from utils.data.raw_datasets import get_raw_dataset, PromptRawDataset

from transformers import PreTrainedTokenizer
from typing import List, Dict, Union, Callable
from utils.global_utils import IGNORE_INDEX, ENS_TOKEN

class SFTDataset(Dataset):
    def __init__(
        self,
        dataset : List
    ):
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index : int):
        return {
            'input_ids' : self.dataset[index]['input_ids'],
            'attention_mask' : self.dataset[index]['attention_mask'],
            'labels' : self.dataset[index]['labels']
        }
    
class RewardDataset(Dataset):
    def __init__(
        self, 
        chosen_dataset : List,
        reject_dataset : List
    ):
        super().__init__()
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self._use_splitted_mask = 'splitted_mask' in chosen_dataset[0].keys()

    def __len__(self):    
        return len(self.chosen_dataset)
    
    def __getitem__(self, index : int) -> List:
        if self._use_splitted_mask:
            return (
                self.chosen_dataset[index]['input_ids'],
                self.chosen_dataset[index]['attention_mask'],
                self.chosen_dataset[index]['splitted_mask'],
                self.reject_dataset[index]['input_ids'],
                self.reject_dataset[index]['attention_mask'],
                self.reject_dataset[index]['splitted_mask']
            )
        else:
            return (
                self.chosen_dataset[index]['input_ids'],
                self.chosen_dataset[index]['attention_mask'],
                self.reject_dataset[index]['input_ids'],
                self.reject_dataset[index]['attention_mask']
            )

class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        if len(data[0]) == 4:
            batch["input_ids"] = torch.cat(
                [f[0] for f in data] + [f[2] for f in data], dim=0
            )
            batch["attention_mask"] = torch.cat(
                [f[1] for f in data] + [f[3] for f in data], dim=0
            )
        elif len(data[0]) == 6:
            batch["input_ids"] = torch.cat(
                [f[0] for f in data] + [f[3] for f in data], dim=0
            )
            batch["attention_mask"] = torch.cat(
                [f[1] for f in data] + [f[4] for f in data], dim=0
            )
            batch["splitted_mask"] = torch.cat(
                [f[2] for f in data] + [f[5] for f in data], dim=0
            )
        else:
            raise ValueError(f"Unsupported batch shape {len(data)}")
        return batch

def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def get_raw_dataset_split_index(
    local_rank,
    output_path : str,
    dataset_name : str,
    seed : int,
    split_name : str,
    data_split : str,
    split_index : int,
    data_size : int,
):
    index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"
    if os.path.isfile(index_file_name):
        print(f'Find cached index file {index_file_name}')
    # reindex each time when using local jsonfile since it's more likely to get modified
    if (not os.path.isfile(index_file_name)) or (dataset_name == "jsonfile"):
        splits = [float(s) for s in data_split.split(",")]
        splits_sum = sum(splits)
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(
                splits_index[index] + int(round(split * float(data_size)))
            )
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert splits_index[-1] == data_size

        shuffle_idx = get_shuffle_idx(seed, data_size)
        for split_i in range(len(splits)):
            shuffle_idx_split_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy"
            shuffle_idx_split = shuffle_idx[
                splits_index[split_i] : splits_index[split_i + 1]
            ]
            np.save(shuffle_idx_split_file_name, shuffle_idx_split, allow_pickle=True)
    
    index = np.load(index_file_name, allow_pickle=True)

    return index.tolist()

def _get_save_name(
    data_path : str,
    data_split : str,
    output_path : str,
    split_index : int,
    seed : int,
    tokenizer : PreTrainedTokenizer,
    max_seq_len : int,
    suffix : str,   
):
    fname = "_".join(data_path)
    tokenizer_name : str = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    if '-Instruct' in tokenizer_name:
        tokenizer_name = tokenizer_name.replace('-Instruct', '')
        output_path = output_path.replace('-Instruct', '')
        
    os.makedirs(output_path, exist_ok=True)
    fname = f"{fname}_split{data_split}_phase{split_index}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}{suffix}"
    fname_no_hash = "_".join(fname.split("/"))
    # for debug
    fname_no_hash = 'HuggingFaceH4_ultrafeedback_binarized_split0,10_phase1_seed1234_tokenizermeta-llama_Llama-3.2-1B_seqlen1024splited'
    
    fname = hashlib.sha256(
        fname_no_hash.encode()
    ).hexdigest()  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    if cache_found:
        print(f'Loading cached dataset file! {output_path}/{fname_no_hash}')
    else:
        print(f'Processing dataset from scratch! {output_path}/{fname_no_hash}')
    buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    torch.distributed.all_reduce(buf_create_cache)

    return output_path, train_fname, eval_fname, buf_create_cache

def _create_sft_dataset(
    subset : Subset,
    raw_dataset : PromptRawDataset,
    tokenizer : PreTrainedTokenizer,
    end_of_conversation_token : str,
    max_seq_len : int,
    **kwargs,    
):
    skip_ENS_token = kwargs.get('skip_ENS_token', True)
    ENS_token = kwargs.get('ENS_token', ENS_TOKEN)
    results = []
    for i, tmp_data in enumerate(subset):
        prompt = raw_dataset.get_prompt(tmp_data)
        chosen_seq = raw_dataset.get_prompt_and_chosen(tmp_data)
        assert chosen_seq is not None
        prompt_tokens = tokenizer(
            prompt,
            max_length = max_seq_len,
            padding = 'do_not_pad',
            truncation = True,
            return_tensors = 'pt',
            add_special_tokens = False
        )
        prompt_length = len(prompt_tokens['input_ids'].flatten())
        
        # if not skip_ENS_token:
        #     if not chosen_seq.endswith(end_of_conversation_token):
        #         chosen_seq += ' ' + end_of_conversation_token
        # else:
        #     chosen_seq = chosen_seq.rstrip(ENS_token) + end_of_conversation_token + ENS_token    
        # if not skip_ENS_token:
        #     chosen_seq = chosen_seq.rstrip(ENS_TOKEN) + end_of_conversation_token + ENS_TOKEN
        chosen_seq = chosen_seq.rstrip(ENS_TOKEN) + end_of_conversation_token + ENS_TOKEN
       
        chosen_tokens = tokenizer(
            chosen_seq,
            max_length = max_seq_len,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt',
            add_special_tokens = False
        )
        
        chosen_tokens['input_ids'] = chosen_tokens['input_ids'].squeeze(0)
        chosen_tokens['attention_mask'] = chosen_tokens['attention_mask'].squeeze(0)
        labels = deepcopy(chosen_tokens['input_ids'])
        if prompt_length == max_seq_len:
            continue
        else:
            # ignore the loss of prompt tokens
            labels[ : prompt_length] = IGNORE_INDEX
        
        # ignore the loss of padding token
        attention_mask = chosen_tokens['attention_mask']
        padding_length = attention_mask.unsqueeze(0).fliplr().argmax(dim = 1)
        if padding_length > 0:
            padding_start_index = attention_mask.shape[0] - padding_length
            labels[padding_start_index : ] = IGNORE_INDEX
        
        # we don't calculate attention score for ENS token
        if skip_ENS_token:
            ENS_token_id = tokenizer.encode(ENS_token, add_special_tokens = False)
            assert len(ENS_token_id) == 1, f'{ENS_token} is not set to be a special token!'
            ENS_token_id = ENS_token_id[0]
            not_ENS_tokens = (chosen_tokens['input_ids'] != ENS_token_id)
            attention_mask = (attention_mask & not_ENS_tokens)
            chosen_tokens['attention_mask'] = attention_mask
            ENS_tokens = (chosen_tokens['input_ids'] == ENS_token_id)
            labels[ENS_tokens] = IGNORE_INDEX

        chosen_tokens['labels'] = labels
        results.append(chosen_tokens)
        if (i + 1) % int(len(subset) // 10) == 0:
            print(f'Processed {100 * (i + 1) / len(subset) }%')
            
    return SFTDataset(
        dataset = results
    )

        
def _create_reward_dataset(
    subset : Subset,
    raw_dataset : PromptRawDataset,
    tokenizer : PreTrainedTokenizer,
    end_of_conversation_token : str,
    max_seq_len : int,
):
    chosen_dataset = []
    reject_dataset = []

    for i, tmp_data in enumerate(subset):
        chosen_sentence = raw_dataset.get_prompt_and_chosen(tmp_data)
        reject_sentence = raw_dataset.get_prompt_and_rejected(tmp_data)
        assert (chosen_sentence is not None) and (reject_sentence is not None)
        chosen_sentence = chosen_sentence.rstrip('\n')
        reject_sentence = reject_sentence.rstrip('\n')
        if not chosen_sentence.endswith(end_of_conversation_token):
            chosen_sentence += ' ' + end_of_conversation_token
        if not reject_sentence.endswith(end_of_conversation_token):
            reject_sentence += ' ' + end_of_conversation_token
        
        chosen_token = tokenizer(
            chosen_sentence,
            max_length = max_seq_len,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt'
        )
        reject_token = tokenizer(
            reject_sentence,
            max_length = max_seq_len,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt'
        )
        chosen_dataset.append(
            dict(
                input_ids = chosen_token['input_ids'],
                attention_mask = chosen_token['attention_mask']
            )
        )
        reject_dataset.append(
            dict(
                input_ids = reject_token['input_ids'],
                attention_mask = reject_token['attention_mask']
            )
        )
    
    return RewardDataset(
        chosen_dataset,
        reject_dataset
    )

def _create_sentence_reward_dataset(
    subset : Subset,
    raw_dataset : PromptRawDataset,
    tokenizer : PreTrainedTokenizer,
    end_of_conversation_token : str,
    max_seq_len : int,
    **kwargs  
):
    from utils.data.spliter import Spliter
    assert 'spliter' in kwargs.keys()
    spliter : Spliter = kwargs['spliter']
    chosen_dataset = []
    reject_dataset = []
    for i, tmp_data in enumerate(subset):
        chosen_seq = raw_dataset.get_prompt_and_chosen(tmp_data)
        reject_seq = raw_dataset.get_prompt_and_rejected(tmp_data)
        chosen_tokens = spliter.tokenization(chosen_seq, max_seq_len)
        reject_tokens = spliter.tokenization(reject_seq, max_seq_len)
        chosen_dataset.append(chosen_tokens)
        reject_dataset.append(reject_tokens)
        if (i + 1) % int(len(subset) // 10) == 0:
            print(f'Processed {100 * (i + 1) / len(subset) }%')
        
    return RewardDataset(
        chosen_dataset = chosen_dataset,
        reject_dataset = reject_dataset
    )

def _create_token_reward_dataset(
    subset : Subset,
    raw_dataset : PromptRawDataset,
    tokenizer : PreTrainedTokenizer,
    end_of_conversation_token : str,
    max_seq_len : int,
    **kwargs  
):
    from utils.data.spliter import Spliter
    assert 'spliter' in kwargs.keys()
    spliter : Spliter = kwargs['spliter']
    chosen_dataset = []
    reject_dataset = []
    for i, tmp_data in enumerate(subset):
        chosen_seq = raw_dataset.get_prompt_and_chosen(tmp_data)
        reject_seq = raw_dataset.get_prompt_and_rejected(tmp_data)
        chosen_tokens = spliter.tokenization_wo_end(chosen_seq, max_seq_len)
        reject_tokens = spliter.tokenization_wo_end(reject_seq, max_seq_len)
        chosen_dataset.append(chosen_tokens)
        reject_dataset.append(reject_tokens)
        if (i + 1) % int(len(subset) // 10) == 0:
            print(f'Processed {100 * (i + 1) / len(subset) }%')
        
    return RewardDataset(
        chosen_dataset = chosen_dataset,
        reject_dataset = reject_dataset
    )


# def _create_sentence_reward_dataset(
#     subset : Subset,
#     raw_dataset : PromptRawDataset,
#     tokenizer : PreTrainedTokenizer,
#     end_of_conversation_token : str,
#     max_seq_len : int,
#     **kwargs
# ):
#     from utils.data.spliter import Spliter
#     from tqdm import tqdm
#     assert 'spliter' in kwargs.keys()
#     spliter : Spliter = kwargs['spliter']
#     chosen_dataset = []
#     reject_dataset = []

#     print(f'***** Sentence split | Size {len(subset)} *****')
#     for i, tmp_data in enumerate(subset):
#         prompt = raw_dataset.get_prompt(tmp_data)
#         chosen_seq = raw_dataset.get_chosen(tmp_data)
#         reject_seq = raw_dataset.get_rejected(tmp_data)
#         try:
#             # split response sequence into sub-sequence
#             chosen_sentences = spliter.split_texts(chosen_seq)[0]
#             reject_sentences = spliter.split_texts(reject_seq)[0]
#             # we don't split prompt anymore (maybe)
#             chosen_seq = prompt + spliter._end_of_sentence_token + chosen_sentences
#             reject_seq = prompt + spliter._end_of_sentence_token + reject_sentences
#             # tokenize
#             chosen_tokens = spliter.tokenization(chosen_seq, max_seq_len)
#             reject_tokens = spliter.tokenization(reject_seq, max_seq_len)
#             chosen_dataset.append(chosen_tokens)
#             reject_dataset.append(reject_tokens)
#         except:
#             print(f'Invalid sequence : \n{chosen_seq}\n{reject_seq}\noccured, which can\'t be splitted!')
#             continue

#         torch.distributed.barrier()

#         # log
#         if (i + 1) % int(len(subset) // 20) == 0:
#             print(f'Processed {100 * (i + 1) / len(subset) }%')

#     return RewardDataset(
#         chosen_dataset = chosen_dataset,
#         reject_dataset = reject_dataset
#     )



def _create_dataset(
    local_rank : int,
    dataset_name : str,
    data_split : str,
    output_path : str,
    split_index : int,
    seed : int,
    tokenizer : PreTrainedTokenizer,
    end_of_conversation_token : str,
    max_seq_len : int,
    create_dataset_fn : Callable,
    **kwargs,
) -> Dataset:
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    train_dataset = raw_dataset.get_train_data()
    train_index = get_raw_dataset_split_index(
        local_rank,
        output_path,
        raw_dataset.dataset_name_clean,
        seed,
        "train",
        data_split,
        split_index,
        len(train_dataset),
    )
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_fn(
        train_dataset,
        raw_dataset,
        tokenizer,
        end_of_conversation_token,
        max_seq_len,
        **kwargs,
    )

    eval_dataset = raw_dataset.get_eval_data()
    eval_index = get_raw_dataset_split_index(
        local_rank,
        output_path,
        raw_dataset.dataset_name_clean,
        seed,
        "eval",
        data_split,
        split_index,
        len(eval_dataset),
    )
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_fn(
        eval_dataset,
        raw_dataset,
        tokenizer,
        end_of_conversation_token,
        max_seq_len,
        **kwargs
    )
    return train_dataset, eval_dataset

def create_dataset(
    local_rank : int,
    data_path : str,
    data_split : str,
    output_path : str,
    split_index : int,
    seed : int,
    tokenizer : PreTrainedTokenizer,
    max_seq_len : int,
    end_of_conversation_token : str = '<|endoftext|>',
    reload : bool = False,
    dataset_type : str = 'reward',
    **kwargs,    
):
    assert dataset_type.lower() in ['sft', 'reward', 'rlhf', 'sentence-reward', 'token-reward']
    create_dataset_fns = {
        'sft' : _create_sft_dataset,
        'reward' : _create_reward_dataset,
        'rlhf' : NotImplementedError,
        'sentence-reward' : _create_sentence_reward_dataset,
        'token-reward' : _create_token_reward_dataset,
    }
    os.makedirs(output_path, exist_ok = True)
    create_fn = create_dataset_fns[dataset_type.lower()]
    if len(data_path) == 1:
        train_set, eval_set = _create_dataset(
            local_rank, data_path[0], data_split,
            output_path, split_index, seed, tokenizer,
            end_of_conversation_token, max_seq_len, create_fn, **kwargs,
        )
    else:
        train_sets, eval_sets = [], []
        train_size, eval_size = 0, 0
        for path in data_path:
            train_set, eval_set = _create_dataset(
                local_rank, path, data_split,
                output_path, split_index, seed, tokenizer,
                end_of_conversation_token, max_seq_len, create_fn, **kwargs
            )
            train_sets.append(train_set)
            eval_sets.append(eval_set)
            train_size += len(train_set)
            eval_size += len(eval_set)
        
        train_set, eval_set = ConcatDataset(train_sets), ConcatDataset(eval_sets)
        shuffle_idx = get_shuffle_idx(seed, train_size)
        train_set = Subset(train_set, shuffle_idx.tolist())
        shuffle_idx = get_shuffle_idx(seed, eval_size)
        eval_set = Subset(eval_set, shuffle_idx.tolist())
    
    torch.distributed.barrier()
    return train_set, eval_set
    
# def create_dataset(
#     local_rank : int,
#     data_path : str,
#     data_split : str,
#     output_path : str,
#     split_index : int,
#     seed : int,
#     tokenizer : PreTrainedTokenizer,
#     max_seq_len : int,
#     end_of_conversation_token : str = '<|endoftext|>',
#     reload : bool = False,
#     dataset_type : str = 'reward',
#     **kwargs,    
# ):
#     assert dataset_type.lower() in ['sft', 'reward', 'rlhf', 'sentence-reward']
#     create_dataset_fns = {
#         'sft' : _create_sft_dataset,
#         'reward' : _create_reward_dataset,
#         'rlhf' : NotImplementedError,
#         'sentence-reward' : _create_sentence_reward_dataset,
#     }
#     # suffix = dataset_type if dataset_type != 'sentence-reward' else 'splited'
#     suffix = '_' + dataset_type.lower()
#     output_path, train_fname, eval_fname, buf_create_cache = _get_save_name(
#         data_path, data_split, output_path, split_index, seed, tokenizer, max_seq_len, suffix
#     )

#     create_fn = create_dataset_fns[dataset_type.lower()]
#     if (local_rank <= 0) and (buf_create_cache.item() != 0 or reload):
#         if (data_path) == 1:
#             train_set, eval_set = _create_dataset(
#                 local_rank, data_path[0], data_split,
#                 output_path, split_index, seed, tokenizer,
#                 end_of_conversation_token, max_seq_len, create_fn, **kwargs,
#             )
#         else:
#             train_sets, eval_sets = [], []
#             train_size, eval_size = 0, 0
#             for path in data_path:
#                 train_set, eval_set = _create_dataset(
#                     local_rank, path, data_split,
#                     output_path, split_index, seed, tokenizer,
#                     end_of_conversation_token, max_seq_len, create_fn, **kwargs
#                 )
#                 train_sets.append(train_set)
#                 eval_sets.append(eval_set)
#                 train_size += len(train_set)
#                 eval_size += len(eval_set)
            
#             train_set, eval_set = ConcatDataset(train_sets), ConcatDataset(eval_sets)
#             shuffle_idx = get_shuffle_idx(seed, train_size)
#             train_set = Subset(train_set, shuffle_idx.tolist())
#             shuffle_idx = get_shuffle_idx(seed, eval_size)
#             eval_set = Subset(eval_set, shuffle_idx.tolist())
        
#         torch.save(train_set, train_fname)
#         torch.save(eval_set, eval_fname)
    
#     torch.distributed.barrier()
#     return torch.load(train_fname), torch.load(eval_fname)