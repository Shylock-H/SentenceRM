# a scripts to process data
from typing import Dict, List, Union
from datasets import (
    load_dataset, 
    load_from_disk, 
    Dataset,
    DatasetDict
)
from transformers import(
    PreTrainedTokenizer,
    AutoTokenizer
)
from tqdm import tqdm
import os

from utils.data.spliter import Spliter
from utils.data.raw_datasets import PromptRawDataset, HuggingFaceH4UltrafeedbackBinarized
from utils.global_utils import SAT_MODEL, ENS_TOKEN

def save_dataset(dataset : Union[Dict, List], save_dir : str, file_name : str, suffix : str = '.json'):
    if isinstance(dataset, List):
        datasets = Dataset.from_list(dataset)
    elif isinstance(dataset, Dict):
        datasets = Dataset.from_dict(dataset)
    else:
        raise NotImplementedError
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = file_name + suffix if suffix is not None else file_name + '.parquet'
    file_path = os.path.join(save_dir, file_name)
    
    if file_path.endswith('csv'):
        datasets.to_csv(file_path)
    elif file_path.endswith('json'):
        datasets.to_json(file_path)
    else:
        datasets.to_parquet(file_path)

    print(f'Save dataset to {file_path}')

    return file_path

def load_local_dataset(save_path : str):
    support_suffixes = ['csv', 'json', 'parquet']
    if os.path.isdir(save_path):
        all_files = list(filter(lambda x : any(suffix in x for suffix in support_suffixes), os.listdir(save_path)))
        dataset_files = [os.path.join(save_path, file) for file in all_files]
    else:
        all_files = save_path.split(os.sep)[-1]
        dataset_files = [save_path]
    
    assert len(dataset_files) > 0
    def load_single_dataset(
        dataset_path : str,          
    ):
        for suffix in support_suffixes:
            if suffix in dataset_path:
                key = dataset_path.split(os.sep)[-1].replace('.' + suffix, '')
                dataset = load_dataset(suffix, data_files = dataset_path)['train']
                break
        
        return key, dataset
 
    dataset_files = [load_single_dataset(dataset_path) for dataset_path in dataset_files]
    if len(dataset_files) == 1:
        return dataset_files[0]
    else:
        datasets = {k : dataset for (k, dataset) in dataset_files}

        return DatasetDict(datasets)

def split_into_sentence(
    raw_dataset : PromptRawDataset,
    tokenizer : PreTrainedTokenizer,
    sat_model : str = SAT_MODEL, 
    end_of_sentence_token : str = ENS_TOKEN,
    max_sentence_num : int = None,
):
    spliter = Spliter(tokenizer, sat_model, end_of_sentence_token, tokenizer.eos_token, max_sentence_num = max_sentence_num)
    spliter.to('cuda:0')
    train_dataset = raw_dataset.get_train_data()
    eval_dataset = raw_dataset.get_eval_data()
    def split(dataset : Dict):
        rets = []
        for tmp_data in tqdm(dataset):
            prompt = raw_dataset.get_prompt(tmp_data)
            chosen_seq = raw_dataset.get_chosen(tmp_data)
            reject_seq = raw_dataset.get_rejected(tmp_data)
            try:
                # split response sequence into sub-sequence
                chosen_sentences = spliter.split_texts(chosen_seq)[0]
                reject_sentences = spliter.split_texts(reject_seq)[0]
                # we don't split prompt anymore (maybe)
                chosen_seq = prompt + spliter._end_of_sentence_token + chosen_sentences
                reject_seq = prompt + spliter._end_of_sentence_token + reject_sentences
                # save
                item = dict(
                    prompt = prompt,
                    chosen_response = chosen_sentences,
                    rejected_response = reject_sentences,
                    chosen = chosen_seq,
                    rejected = reject_seq
                )
                rets.append(item)
            except:
                print(f'\nInvalid sequence : \nchosen\n{chosen_seq}\nreject\n{reject_seq}\noccured, which can\'t be splitted!')
                assert 0

        return rets
        
    train_list = split(train_dataset)
    eval_list = split(eval_dataset)
    
    return train_list, eval_list

def split_into_sentence_batch_mode(
    raw_dataset : PromptRawDataset,
    tokenizer : PreTrainedTokenizer,
    sat_model : str = SAT_MODEL, 
    end_of_sentence_token : str = ENS_TOKEN,
    max_sentence_num : int = None,
    batch_size : int = 64,
):
    spliter = Spliter(tokenizer, sat_model, end_of_sentence_token, tokenizer.eos_token, max_sentence_num = max_sentence_num)
    spliter.to('cuda:0')
    train_dataset = raw_dataset.get_train_data()
    eval_dataset = raw_dataset.get_eval_data()
    def split(dataset : Dict):
        rets = []
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i : i + batch_size]
            list_length = len(next(iter(batch.values())))
            list_batch = [
                {key: batch[key][i] for key in batch}
                for i in range(list_length)
            ]
            prompts = [raw_dataset.get_prompt(item) for item in list_batch]
            chosen_seqs = [raw_dataset.get_chosen(item) for item in list_batch]
            reject_seqs = [raw_dataset.get_rejected(item) for item in list_batch]
            chosen_sentences = spliter.split_texts(chosen_seqs)
            reject_sentences = spliter.split_texts(reject_seqs)
            print(chosen_sentences)
            assert 0

        # for tmp_data in tqdm(dataset):
        #     prompt = raw_dataset.get_prompt(tmp_data)
        #     chosen_seq = raw_dataset.get_chosen(tmp_data)
        #     reject_seq = raw_dataset.get_rejected(tmp_data)
        #     try:
        #         # split response sequence into sub-sequence
        #         chosen_sentences = spliter.split_texts(chosen_seq)[0]
        #         reject_sentences = spliter.split_texts(reject_seq)[0]
        #         # we don't split prompt anymore (maybe)
        #         chosen_seq = prompt + spliter._end_of_sentence_token + chosen_sentences
        #         reject_seq = prompt + spliter._end_of_sentence_token + reject_sentences
        #         # save
        #         item = dict(
        #             prompt = prompt,
        #             chosen_response = chosen_sentences,
        #             rejected_response = reject_sentences,
        #             chosen = chosen_seq,
        #             rejected = reject_seq
        #         )
        #         rets.append(item)
        #     except:
        #         print(f'\nInvalid sequence : \nchosen\n{chosen_seq}\nreject\n{reject_seq}\noccured, which can\'t be splitted!')
        #         assert 0

        # return rets
        
    train_list = split(train_dataset)
    eval_list = split(eval_dataset)
    
    return train_list, eval_list

def merge_dataset(dataset_paths : List[str]):
    train_list, eval_list = [], []
    for path in dataset_paths:
        dataset = load_local_dataset(path)
        train_set, eval_set = dataset['train'], dataset['eval']
        train_list += [data for data in train_set]
        eval_list += [data for data in eval_set]
    
    return train_list, eval_list

if __name__ == '__main__':
    cur_dir = os.getcwd()
    save_dir = 'dataset_files'
    output_dir = os.path.join(cur_dir, save_dir, f'ultrafeedback_split_3l')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    dataset_name = 'HuggingFaceH4/ultrafeedback_binarized'
    raw_dataset = HuggingFaceH4UltrafeedbackBinarized(output_dir, 1234, 0, dataset_name)
    train_set, eval_set = split_into_sentence_batch_mode(raw_dataset, tokenizer, SAT_MODEL, ENS_TOKEN, None, 2)
    save_dataset(train_set, output_dir, 'train', None)
    save_dataset(eval_set, output_dir, 'eval', None)