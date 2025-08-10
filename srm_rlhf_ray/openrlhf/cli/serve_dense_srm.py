import argparse
import re
import math
import time
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer

from openrlhf.utils.logging_utils import init_logger
from openrlhf.models.spliter import create_spliter, SAT_MODEL, ENS_TOKEN
from openrlhf.models.srm import load_srm_model
logger = init_logger(__name__)


class RewardModelProxy:
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.reward_pretrain,
            use_fast=not args.disable_fast_tokenizer
        )
        self.reward_model = load_srm_model(
            args.reward_pretrain,
            self.tokenizer,
            bf16=args.bf16
        ).to(torch.cuda.current_device())
        self.reward_model.eval()

        self.max_length = args.max_len
        self.batch_size = args.batch_size
        self.normalize_reward = args.normalize_reward

        self.spliter =  create_spliter(
            self.tokenizer, 
            model_name = SAT_MODEL, 
            end_sentence_token = ENS_TOKEN, 
            end_text_token = self.tokenizer.eos_token
        )

    def get_reward(self, raw_prompts: list[str], answers: list[str]):
        start_time = time.time()
        template = ['\n\nHuman: ', '\n\nAssistant: ']
        processed_queries = []
        # remove pad_token
        for i in range(len(raw_prompts)):
            if not answers[i].endswith(self.spliter._end_of_sentence_token):
                answers[i] += self.spliter._end_of_sentence_token
            if answers[i][0] == ' ':
                answers[i] = answers[i][1:] # remove the first space

            human_prefix = '\n\nHuman: '
            assistant_prefix = '\n\nAssistant: '
            max_prompt_len = 1024 // 2  # 512

            prompt = human_prefix + raw_prompts[i]
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(prompt_ids) > max_prompt_len:
                prefix_ids = self.tokenizer.encode(human_prefix, add_special_tokens=False)
                max_raw_prompt_len = max_prompt_len - len(prefix_ids)
                raw_prompt_ids = self.tokenizer.encode(raw_prompts[i], add_special_tokens=False)
                truncated_raw_prompt_ids = raw_prompt_ids[-max_raw_prompt_len:]
                truncated_raw_prompt = self.tokenizer.decode(truncated_raw_prompt_ids, skip_special_tokens=True)
                prompt = human_prefix + truncated_raw_prompt

            processed_query = prompt + assistant_prefix + self.spliter._end_of_sentence_token + answers[i]
            processed_queries.append(processed_query)
        logger.info(f"raw_prompts[0]: {repr(raw_prompts[0])}")
        logger.info(f"answers[0]: {repr(answers[0])}")
        logger.info(f"processed_queries[0]: {repr(processed_queries[0])}")

        inputs = [] # return list of list of scores, where each 
        with torch.no_grad():
            for i in range(0, len(processed_queries)):
                input_tokens = self.spliter.tokenization_w_end(processed_queries[i], self.max_length)
                inputs.append(input_tokens)
        time1 = time.time()
        logger.info(f"tokenization time: {time1 - start_time}")
        
        batch_size = self.batch_size or len(inputs)  # 如果没设batch_size就全量推理
        num_batches = math.ceil(len(inputs) / batch_size)
        all_rewards = []
        all_sentence_rewards = []

        with torch.no_grad():
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, len(inputs))
                batch_inputs = inputs[start:end]
                ks = list(batch_inputs[-1].keys())
                model_inputs = {}
                for k in ks:
                    model_inputs[k] = torch.stack([res[k] for res in batch_inputs]).to(torch.cuda.current_device()).view(len(batch_inputs), self.max_length)
                res = self.reward_model.forward_value(**model_inputs)
                rewards = res['rewards']
                sentence_rewards = res['sentence_rewards']
                if isinstance(rewards, torch.Tensor):
                    rewards = rewards.tolist()
                all_rewards.extend(rewards)
                all_sentence_rewards.extend(sentence_rewards)
        time2 = time.time()
        logger.info(f"forward time: {time2 - time1}")
        return all_rewards, all_sentence_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normalization")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default=2048)

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    # server
    reward_model = RewardModelProxy(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        raw_prompts = data.get("raw_prompts", [])
        answers = data.get("answers", [])
        
        rewards, sentence_rewards = reward_model.get_reward(
            raw_prompts, 
            answers
        )
        # Convert tensor to list for JSON serialization
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.tolist()
        for i in range(len(sentence_rewards)):
            if isinstance(sentence_rewards[i], torch.Tensor):
                sentence_rewards[i] = sentence_rewards[i].tolist()

        result = {"rewards": rewards, "sentence_rewards": sentence_rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")