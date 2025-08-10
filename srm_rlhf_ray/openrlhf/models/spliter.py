import warnings
import torch
import torch.nn as nn
import numpy as np
import polars as pl                             
import pyarrow as pa
import pyarrow.compute as pc
import typing as tp
from dataclasses import dataclass, field
from stopes.utils.arrow_utils import (          
    apply_on_nested_array,
)                                         
from operator import itemgetter
from typing import Union, List, Dict
from wtpsplit import SaT, indices_to_sentences
from transformers import PreTrainedModel, PreTrainedTokenizer

IGNORE_INDEX = -100
SAT_MODEL = 'sat-3l'
ENS_TOKEN = '<END>'
SPLIT_THRESHOLD = None

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

def convert_token_to_id(token : str, tokenizer : PreTrainedTokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")

@dataclass
class SentenceSpliterConfig:
    model_name: str = "sat-3l"
    sentence_suffix: str = "_sentences"
    sentence_threshold: float = 0.2
    max_sentence_len: int = 256
    min_text_length: int = 10
    min_unique_chars: int = 0
    fallback_separators: List[str] = field(
        default_factory = lambda: [
            "...",
            "\n",
            "!",
            "?",
            ";",
            ":",
            ".",
            ",",
            "\t",
            " ",
        ]
    )
    device: str = "cpu"
    remove_whitespace_before_inference: bool = False
    batch_size: int = 256
    block_size: int = 256
    stride: int = 256
    outer_batch_size: int = 1024
    verbose: bool = False
    pad_last_batch: bool = False

class Spliter:
    """
    A module to split text into sentences
    """
    def __init__(
        self, 
        tokenizer : PreTrainedTokenizer,
        model_name : str = SAT_MODEL,
        end_of_sentence_token : str = ENS_TOKEN,
        end_of_conversation_token : str = '</s>',
        max_sentence_num : int = None
    ):
        """
        Initialize the spliter.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to use.
            model_name (str): The name of the model to use.
            end_of_sentence_token (str): The token to use to indicate the end of a sentence. Only use to auxiliary the model to split the text. (default: '<END>')
            end_of_conversation_token (str): The token to use to indicate the end of a conversation. (default: '</s>')
            max_sentence_num (int): The maximum number of sentences to split. (default: None)
        """
        spliter_config = SentenceSpliterConfig(
            model_name = model_name,
            sentence_threshold = SPLIT_THRESHOLD
        )
        self.model = SaT(SAT_MODEL)

        self.config = spliter_config
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({'additional_special_tokens': [end_of_sentence_token]})

        self._end_of_sentence_token = end_of_sentence_token

        self._end_of_text_token = end_of_conversation_token
        self._splited_token_id = self.tokenizer.encode(end_of_sentence_token)[-1]
        self._max_sentence_num = 10000 if max_sentence_num is None else max_sentence_num
    
    def to(self, device : str):
        self.model.half().to(device)
    
    def _resplit_long_sentences(self, col: pa.Array) -> list[list[str]]:
        mask = pc.greater_equal(pc.utf8_length(col), self.config.max_sentence_len)
        texts_to_resplit = col.filter(mask).to_pandas().to_list()

        resplit_sentences = []
        for text, probs in zip(
            texts_to_resplit,
            self.model.predict_proba(
                texts_to_resplit,
                stride = self.config.stride,
                block_size = self.config.block_size,
                batch_size = self.config.batch_size,
                pad_last_batch = self.config.pad_last_batch,
                remove_whitespace_before_inference = self.config.remove_whitespace_before_inference,
                outer_batch_size = self.config.outer_batch_size,
                verbose = self.config.verbose
            ),
        ):
            nb_split = round(len(probs) / self.config.max_sentence_len) + 1
            sentence_threshold = np.partition(probs, -nb_split)[-nb_split]
            sentences = indices_to_sentences(
                text,
                np.where(probs >= sentence_threshold)[0],
                strip_whitespace = False,
            )
            resplit_sentences.append(sentences)

        # if not, hard resplit with some separators
        def _resplit(raw_sentences):
            for separator in self.config.fallback_separators:
                new_sentences = []
                for sent in raw_sentences:
                    for subchunk in self._split_with_max_length(
                        sent, max_length = self.config.max_sentence_len, sep = separator
                    ):
                        new_sentences.append(subchunk)
            return new_sentences

        np_mask = mask.to_pandas().to_numpy()
        full_text = col.to_pandas().to_list()

        output_sentences = []
        j = 0
        for i, text in enumerate(full_text):
            if np_mask[i]:
                output_sentences.append(_resplit(resplit_sentences[j]))
                j += 1
            else:
                output_sentences.append([text])

        return pa.array(output_sentences, type=pa.list_(pa.string()))
    
    def _split_with_max_length(self, text: str, max_length: int, sep: str) -> List[str]:
        words = text.split(sep)
        result = []
        current_piece = ""

        for i, word in enumerate(words[:-1]):
            # Append separator back to each word except the last
            word += sep
            if len(current_piece) + len(word) <= max_length:
                current_piece += word
            else:
                if current_piece:
                    result.append(current_piece)
                current_piece = word

        # Handle the last word separately to avoid adding an extra separator
        last_word = words[-1]
        if len(current_piece) + len(last_word) <= max_length:
            current_piece += last_word
        else:
            if current_piece:
                result.append(current_piece)
            current_piece = last_word

        if current_piece:
            result.append(current_piece)

        return result
    
    def _split_src_texts(
        self,
        texts : Union[str, List[str], torch.Tensor],
    ):
        long_texts = [t for t in texts if len(t) > self.config.min_text_length]
        keep_texts = [
            (idx, t)
            for idx, t in enumerate(texts)
            if len(t) <= self.config.min_text_length
        ]
        outputs = self.model.split(
            long_texts,
            threshold = self.config.sentence_threshold,
            stride = self.config.stride,
            block_size = self.config.block_size,
            batch_size = self.config.batch_size,
            pad_last_batch = self.config.pad_last_batch,
            remove_whitespace_before_inference = self.config.remove_whitespace_before_inference,
            outer_batch_size = self.config.outer_batch_size,
            verbose = self.config.verbose
        )
        sentences = []
        for output in outputs:
            # sentences.append([s.strip() for s in output if s.strip()])
            sentences.append([s for s in output])
        for idx, text in keep_texts:
            sentences.insert(idx, [text])
        
        sentences = pa.array(sentences, type = pa.list_(pa.string()))

        list_texts = apply_on_nested_array(self._resplit_long_sentences, sentences)
        reflatten_texts = pl.from_arrow(list_texts).list.eval(pl.element().explode())  # type: ignore
        
        return reflatten_texts
    
    def _split_tensor_texts(
        self,
        texts : torch.Tensor,
    ):
        """
        Split text in tensor format into the combination of sentences.

        """
        src_texts = self.tokenizer.batch_decode(texts, skip_special_tokens = True)
        # NOTE : remove the BOS and EOS token, if it exists
        eos_token, bos_token = self.tokenizer.eos_token, self.tokenizer.bos_token
        processed_texts = [text.lstrip(bos_token).rstrip(eos_token) for text in src_texts]

        return self._split_src_texts(processed_texts)
    
    def _merge(self, splited_texts : List[str]):
        rets = []
        for splited_text in splited_texts:
            new_text = ''
            for idx, sentence in enumerate(splited_text):
                if sentence.endswith(' ') and sentence != ' ' and idx != len(splited_text) - 1:
                    sentence = sentence[ : -1]
                    new_text += (sentence + self._end_of_sentence_token + ' ')
                else:
                    new_text += (sentence + self._end_of_sentence_token)
            
            rets.append(new_text)
        
        return rets
    
    def _correct_merge(self, splited_texts : List[str]):
        rets = []
        for splited_text in splited_texts:
            # 新增逻辑：合并会被编码成一个token的相邻句子
            merged_sentences = []
            i = 0
            while i < len(splited_text):
                current_sentence = splited_text[i]
                # 检查是否有下一个句子
                if i < len(splited_text) - 1:
                    next_sentence = splited_text[i + 1]
                    # 获取当前句子最后一个单词和下一个句子第一个单词
                    current_last_word = current_sentence.rstrip().split()[-1] if current_sentence.strip() else ""
                    next_first_word = next_sentence.lstrip().split()[0] if next_sentence.strip() else ""
                    # 拼接
                    concat = current_last_word + next_first_word
                    # 用tokenizer编码
                    if current_last_word and next_first_word and hasattr(self, "tokenizer"):
                        tokens = self.tokenizer.encode(concat, add_special_tokens=False)
                        if len(tokens) == 1:
                            # 合并这两个句子
                            current_sentence = current_sentence + next_sentence
                            i += 1  # 跳过下一个句子
                merged_sentences.append(current_sentence)
                i += 1

            # 下面是原有的merge逻辑
            new_text = ''
            for idx, sentence in enumerate(merged_sentences):
                if sentence.endswith(' ') and sentence != ' ' and idx != len(merged_sentences) - 1:
                    sentence = sentence[ : -1]
                    new_text += (sentence + self._end_of_sentence_token + ' ')
                else:
                    new_text += (sentence + self._end_of_sentence_token)
            rets.append(new_text)
        return rets
    
    def split_texts(
        self,
        texts : Union[str, List[str], torch.Tensor],  
    ):
        if isinstance(texts, torch.Tensor):
            assert len(texts.shape) == 2
            rets = self._split_tensor_texts(texts)
        else:
            texts = [texts] if isinstance(texts, str) else texts
            rets = self._split_src_texts(texts)
        rets = self._merge(rets)
        return rets
    
    def tokenization_w_end(
        self,
        texts : Union[str, List[str]],
        max_seq_len : int,
    ):
        input_ids, attention_masks, splitted_masks = [], [], []
        texts = texts if isinstance(texts, List) else [texts]
        for text in texts: 
            # NOTE : consider more about the position of <END> and <EOS>
            assert text.endswith(self._end_of_sentence_token)
            text = text[:-len(self._end_of_sentence_token)] + self._end_of_text_token + self._end_of_sentence_token
            result = self.tokenizer(
                text,
                max_length = max_seq_len,
                padding = 'max_length',
                truncation = True,
                return_tensors = 'pt',
                add_special_tokens = False
            )
            
            input_id, attention_mask = itemgetter('input_ids', 'attention_mask')(result)
            # NOTE : when text is truncated, we should mask sure the final non-padding token be <END>
            final_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim = 1, keepdim = True)
            final_indices = final_indices.item()
            input_id[0][final_indices] = self._splited_token_id

            valid_tokens = (input_id != self._splited_token_id)
            # NOTE : attention <END> or not?
            # attention_mask = (attention_mask & valid_tokens)
            splitted_tokens = (input_id == self._splited_token_id).bool().long()

            input_ids.append(input_id.squeeze(0))
            attention_masks.append(attention_mask.squeeze(0))
            splitted_masks.append(splitted_tokens.squeeze(0))
        
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        splitted_masks = torch.stack(splitted_masks)

        return dict(
            input_ids = input_ids,
            attention_mask = attention_masks,
            splitted_mask = splitted_masks
        )

    def tokenization_wo_end(
        self,
        processed_texts : Union[str, List[str]],
        max_seq_len : int=1024,
    ):
        input_ids, attention_masks, splitted_masks = [], [], []

        processed_texts = processed_texts if isinstance(processed_texts, List) else [processed_texts]
        src_texts = [text.replace(self._end_of_sentence_token, '') for text in processed_texts]

        for text, processed_text in zip(src_texts, processed_texts): 
            if not text.endswith(self._end_of_text_token):
                text = text + self._end_of_text_token
            assert processed_text.endswith(self._end_of_sentence_token)
            processed_text = processed_text[:-len(self._end_of_sentence_token)] + self._end_of_text_token + self._end_of_sentence_token

            result = self.tokenizer(
                text,
                max_length = max_seq_len,
                padding = 'max_length',
                truncation = True,
                return_tensors = 'pt',
                add_special_tokens = False
            )
            
            input_id, attention_mask = itemgetter('input_ids', 'attention_mask')(result)
            eos_indice = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim = 1, keepdim = True)

            split_mask, _ = self.find_split_position(text, processed_text, self._end_of_sentence_token)
            split_mask = split_mask[:max_seq_len]
            split_mask[eos_indice] = True
            split_mask = torch.cat([split_mask, torch.zeros(max_seq_len - len(split_mask))], dim = 0)

            input_ids.append(input_id.squeeze(0))
            attention_masks.append(attention_mask.squeeze(0))
            splitted_masks.append(split_mask.squeeze(0))
        
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        splitted_masks = torch.stack(splitted_masks)

        return dict(
            input_ids = input_ids,
            attention_mask = attention_masks,
            splitted_mask = splitted_masks
        )

    def find_split_position(self, src_text: str, split_text: str, split_token: str = '<END>'):
        """
        Find the split position of the split_text in the src_text.
        Args:
            src_text (str): The source text to be split.
            split_text (str): The text to be split.
            split_token (str): The token to split the text. (default: '<END>')
        Returns:
            mask (torch.Tensor): The mask of the split text.
            split_tokens (List[str]): The tokens of the split text.
        """
        encoding = self.tokenizer(
            src_text,
            add_special_tokens=False,
            return_offsets_mapping=True
        )
        token_ids = encoding['input_ids']
        offset_mapping = encoding['offset_mapping']
        
        split_sentences = split_text.split(split_token)
        if len(split_sentences[-1]) == 0:
            split_sentences = split_sentences[:-1]
        
        current_pos = 0
        char_positions = []
        remaining_text = src_text
        
        for sentence in split_sentences:
            if not sentence:
                continue
            pos = remaining_text.find(sentence, current_pos)
            if pos == -1:
                continue
            end_pos = pos + len(sentence)
            char_positions.append(end_pos)
            current_pos = end_pos
        
        split_indices = []
        for char_pos in char_positions:
            for idx, (start, end) in enumerate(offset_mapping):
                if end >= char_pos:
                    split_indices.append(idx)
                    break
        
        mask = torch.zeros(len(token_ids))
        split_indices_tensor = torch.tensor(split_indices, dtype=torch.long)
        
        if len(split_indices) != len(split_sentences):
            print(f'{len(split_indices)} != {len(split_sentences)}')
            print(f'src_text: {repr(src_text)}')
            print(f'split_text: {repr(split_text)}')

        mask[split_indices_tensor] = 1
        
        split_tokens = [self.tokenizer.decode([token_ids[idx]], skip_special_tokens=False) for idx in split_indices]
        
        return mask, split_tokens
