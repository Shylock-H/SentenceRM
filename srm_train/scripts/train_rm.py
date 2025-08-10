import sys
import argparse, os, json, math
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import torch.distributed
from trainer import RMTrainer
from utils.util import get_strategy, get_tokenizer, get_reward_model, get_sentence_reward_model
from transformers import SchedulerType, get_scheduler
import deepspeed, torch

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--data_path",
        nargs="*",
        default=["Dahoas/rm-static"],
        help="Path to the training dataset. Accepted format:"
        "1) a single data path, 2) multiple datasets in the"
        "form: dataset1-path dataset2-path ...",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="2,4,4",
        help="Comma-separated list of proportions for training"
        "phase 1, 2, and 3 data. For example the split `2,4,4`"
        "will use 60%% of data for phase 1, 20%% for phase 2"
        "and 20%% for phase 3.",
    )
    parser.add_argument(
        "--data_split_index",
        type = int,
        default = 1
    )
    parser.add_argument(
        "--data_output_path",
        type=str,
        default="/tmp/data_files/",
        help="Where to store the data-related files such as shuffle index.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--train_batch_size",
        type = int,
        default = 32,
        help = "Batch size to update model one times"
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--seed", type=int, default=2023, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--data_seed", type=int, default=1234, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for Actor model.",
    )
    parser.add_argument(
        "--disable_dropout",
        action="store_true",
        help="Disable the dropout of the model.",
    )
    # LoRA configs
    parser.add_argument("--lora_rank", type = int, default=0)
    parser.add_argument("--lora_alpha", type = int, default = 16)
    parser.add_argument("--target_modules", type = str, nargs = "*", default = None)
    parser.add_argument("--lora_dropout", type = float, default = 0)
    # deepspeed features
    parser.add_argument(
        "--offload", action="store_true", help="Enable ZeRO Offload techniques."
    )
    parser.add_argument("--bf16", action="store_true", help="Enable bf16.")
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model (and clones).",
    )
    ## Print loss
    parser.add_argument(
        "--print_loss", action="store_true", help="Prints loss at each step."
    )
    ## Tensorboard logging
    parser.add_argument(
        "--enable_tensorboard", action="store_true", help="Enable tensorboard logging"
    )
    parser.add_argument("--tensorboard_path", type=str, default="step2_tensorboard")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    args.tensorboard_path = args.output_dir
    strategy = get_strategy(args)
    strategy.set_seed(args.seed)
    strategy.setup_distributed()
    ds_config = strategy.get_ds_train_config()
    torch.distributed.barrier()
    # backup
    args.global_rank = torch.distributed.get_rank()
    if args.global_rank == 0:
        with open(
            os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8"
        ) as f:
            for key, value in args.__dict__.items():
                json.dump({key: value}, f, ensure_ascii=False)
                f.write("\n")
        # save_code(args.output_dir)
    
    tokenizer = get_tokenizer(args.model_name_or_path)
    rm_model = get_reward_model(args.model_name_or_path, strategy, tokenizer)
    if args.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()

    # prepare dataset
    train_dataloader, eval_dataloader = strategy.setup_dataloader(tokenizer, dataset_type = 'reward')
    optimizer = strategy.create_optimizer(rm_model, **vars(args))
    lr_scheduler = get_scheduler(
        name = args.lr_scheduler_type,
        optimizer = optimizer,
        num_warmup_steps = args.num_warmup_steps,
        num_training_steps = args.num_train_epochs * math.ceil(len(train_dataloader) / strategy.accumulated_gradient)
    )
    rm_model, optimizer, lr_scheduler = strategy.ds_init_train_model(rm_model, optimizer, lr_scheduler, ds_config)
    trainer = RMTrainer(
        model = rm_model,
        strategy = strategy,
        optimizer = optimizer,
        train_dataloader = train_dataloader,
        eval_dataloader = eval_dataloader,
        scheduler = lr_scheduler,
        tokenizer = tokenizer,
        max_norm = strategy.max_norm
    )
    trainer.train()

if __name__ == '__main__':
    main()

