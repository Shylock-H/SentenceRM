from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from datasets import DatasetDict
import os
from utils.global_utils import LOCAL_DATASET_DIR

class PromptRawDataset(object):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        if not os.path.exists(os.path.join(LOCAL_DATASET_DIR, dataset_name)):
            self.raw_datasets = load_dataset(dataset_name)

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample) -> str:
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample) -> str:
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample) -> str:
        return

    def get_prompt_and_chosen(self, sample) -> str:
        return

    def get_prompt_and_rejected(self, sample) -> str:
        return


def get_raw_dataset(dataset_name: str, output_path: str, seed: int, local_rank: int):
    if "Anthropic/hh-rlhf" in dataset_name:
        return AnthropicsHhrlhfDataset(output_path, seed, local_rank, dataset_name)
    elif "HuggingFaceH4/ultrafeedback_binarized" in dataset_name:
        return HuggingFaceH4UltrafeedbackBinarized(output_path, seed, local_rank, dataset_name)
    elif 'ultrafeedback_split' in dataset_name:
        return UltrafeedbackSplit(output_path, seed, local_rank, dataset_name)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )

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


class AnthropicsHhrlhfDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Anthropic/hh-rlhf"
        self.dataset_name_clean = "Anthropic_hh-rlhf"
        print("Preprocessing train Data")
        self.raw_datasets["train"] = self.preprocess_dataset(self.raw_datasets["train"])
        print("Preprocessing eval Data")
        self.raw_datasets["test"] = self.preprocess_dataset(self.raw_datasets["test"])

    def preprocess_dataset(self, dataset):
        def extract_anthropic_prompt(prompt_and_response):
            """Extract the anthropic prompt from a prompt and response pair."""
            search_term = "\n\nAssistant:"
            search_term_idx = prompt_and_response.rfind(search_term)
            assert (
                search_term_idx != -1
            ), f"Prompt and response does not contain '{search_term}'"
            return prompt_and_response[: search_term_idx + len(search_term)]

        def split_prompt_and_responses(ex):
            prompt = extract_anthropic_prompt(ex["chosen"])
            chosen_response = ex["chosen"][len(prompt) :]
            rejected_response = ex["rejected"][len(prompt) :]
            return prompt, chosen_response, rejected_response

        data = []
        for row in tqdm(dataset, desc="Processing HH"):
            prompt, chosen, rejected = split_prompt_and_responses(row)
            data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

        return data

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample["prompt"]

    def get_chosen(self, sample):
        return sample["chosen"]

    def get_rejected(self, sample):
        return sample["rejected"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"] + sample["chosen"]

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"] + sample["rejected"]

class HelpSteerDataset(PromptRawDataset):
    pass


class HuggingFaceH4UltrafeedbackBinarized(PromptRawDataset):
    """
    Raw h4-ultrafeedback-binarized dataset is saved in the following structure:
    {
           {
               "prompt": "Which books are still banned in Canada?",
               "chosen": [
                   {
                       "content": "Which books are still banned in Canada?",
                       "role": "user"
                   },
                   {
                       "content": "As a responsible and ethical AI assistant, I must inform you that I cannot provide a list of banned books in Canada, as there are no books that are currently banned in the country. Canada has a strong commitment to freedom of expression and the right to access information, and there are no laws in place that restrict the publication or distribution of books.\n",
                       "role": "assistant"
                   }
               ],
               "messages": [
                   {
                       "content": "Which books are still banned in Canada?",
                       "role": "user"
                   },
                   {
                       "content": "As a responsible and ethical AI assistant, I must inform you that I cannot provide a list of banned books in Canada, as there are no books that are currently banned in the country. Canada has a strong commitment to freedom of expression and the right to access information, and there are no laws in place that restrict the publication or distribution of books.\n",
                       "role": "assistant"
                   }
               ],

               "prompt_id": "aeccf551d9ba42fdf5f2044de43b8ce6e360fb523ace428317b81d804594e090",
               "rejected": [
                   {
                       "content": "Which books are still banned in Canada?",
                       "role": "user"},
                   {
                       "content": "According to the Canadian Government’s Ban Affront website, there are still several books that are banned in Canada. These include The Begum’s Millionaire, The Education of Little Tree, The Harry Potter series, Lolita, 1984, and Lady Chatterley’s Lover. Some of these books are considered inaccessible due to their age, while others are still legally banned in certain parts of the country.",
                       "role": "assistant"
                       }
                   ],
               "score_chosen": 8.0,
               "score_rejected": 5.0
           }

    }
    """

    def __init__(self, output_path, seed, local_rank, dataset_name, debug = False):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        self.dataset_name = "HuggingFaceH4/ultrafeedback_binarized"
        self.dataset_name_clean = "HuggingFaceH4_ultrafeedback_binarized"
        self.raw_datasets = load_dataset(dataset_name)
        self.debug = debug

    def get_train_data(self):
        if self.debug:
            return self.raw_datasets["train_prefs"].select(range(20))
        else:
            return self.raw_datasets["train_prefs"]

    def get_eval_data(self):
        if self.debug:
            return self.raw_datasets["test_prefs"].select(range(20))
        else:
            return self.raw_datasets["test_prefs"]

    def get_prompt(self, sample):
        # TODO: ZXQ change the prompt, not sure if it is correct
        return "\n\nHuman: " + sample["prompt"] + "\n\nAssistant:"

    def get_chosen(self, sample):
        return ' ' + sample["chosen"][1]["content"]

    def get_rejected(self, sample):
        return ' ' + sample["rejected"][1]["content"]

    def get_prompt_and_chosen(self, sample):
        return (
            "\n\nHuman: "
            + sample["prompt"]
            + "\n\nAssistant:"
            + ' ' + sample["chosen"][1]["content"]
        )

    def get_prompt_and_rejected(self, sample):
        return (
            "\n\nHuman: "
            + sample["prompt"]
            + "\n\nAssistant:"
            + ' ' + sample["rejected"][1]["content"]
        )

class UltrafeedbackSplit(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        self.dataset_name = dataset_name
        self.dataset_name_clean = dataset_name
        save_dir = os.path.join(LOCAL_DATASET_DIR, self.dataset_name)
        print(f'> load dataset from {save_dir}') 
        self.raw_datasets = load_local_dataset(save_dir)
    
    def get_train_data(self):
        return self.raw_datasets['train']
    
    def get_eval_data(self):
        return self.raw_datasets['eval']
    
    def get_prompt(self, sample):
        return sample['prompt']
    
    def get_chosen(self, sample):
        return sample['chosen_response']
    
    def get_rejected(self, sample):
        return sample['rejected_response']
    
    def get_prompt_and_chosen(self, sample):
        return sample['chosen']
    
    def get_prompt_and_rejected(self, sample):
        return sample['rejected']