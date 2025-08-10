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
        pass
    def get_reward(self, raw_prompts: list[str], answers: list[str]):
        all_rewards = [0.0] * len(raw_prompts)
        return all_rewards

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
        
        rewards = reward_model.get_reward(
            raw_prompts, 
            answers
        )
        # Convert tensor to list for JSON serialization
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.tolist()

        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")