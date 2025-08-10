from abc import ABC
import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from operator import itemgetter
from model import Actor
from utils.strategy import Strategy
from utils.global_utils import IGNORE_INDEX

from typing import List, Dict

class SFTTrainer(ABC):
    def __init__(
        self,
        model : Actor,
        strategy : Strategy,
        optimizer : Optimizer,
        train_dataloader : DataLoader,
        eval_dataloader : DataLoader,
        scheduler : LRScheduler,
        tokenizer : PreTrainedTokenizer,
        max_norm : float = 0.5,
        pretrained_mode : bool = False,             
    ):
        super().__init__()
        self.strategy = strategy
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.pretrained_mode = pretrained_mode
    
    def _sft_loss(self, logits : torch.Tensor, labels : torch.Tensor):
        inputs = logits[ : , : -1, : ].contiguous().view(-1, logits.size(-1))
        targets = labels[ : , 1 : ].contiguous().view(-1)

        return nn.functional.cross_entropy(inputs, targets, ignore_index = IGNORE_INDEX)
    
    def _log_to_tb(self, logs : Dict, prefix : str = ''):
        if self.model.monitor.enabled and self.model.global_rank == 0:
            events = []
            for k, v in logs.items():
                if isinstance(v, torch.Tensor):
                    v = v.mean().item()
                elif isinstance(v, List) or isinstance(v, np.ndarray):
                    v = np.mean(v).item()
                k = f'{prefix}/{k}' if len(k) > 0 else k
                events.append((k, v, self.model.global_samples))
            self.model.monitor.write_events(events)
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        losses = 0.0
        for step, batch in enumerate(self.eval_dataloader):
            if torch.isnan(batch['input_ids']).int().sum() > 0:
                self.strategy.print(f'Nan occured in input tensors!!!')
                continue
            batch = self.strategy.to_device(batch)
            # outputs = self.model(batch['input_ids'], attention_mask = batch['attention_mask'])
            # labels = batch['labels']
            # loss = self._sft_loss(outputs['logits'], labels)
            outputs = self.model(**batch)
            loss = outputs['loss']
            losses += loss.item()
        
        losses = losses / (step + 1)
        try:
            ppl = np.exp(losses)
        except OverflowError:
            ppl = float('inf')
        try:
            ppl = self.strategy.all_reduce(ppl, 'mean')
        except:
            pass

        return dict(
            loss = loss,
            ppl = ppl
        )

    def _train_epoch(self, epoch : int):
        for step, batch in enumerate(self.train_dataloader):
            batch = self.strategy.to_device(batch)
            # outputs = self.model(batch['input_ids'], attention_mask = batch['attention_mask'])
            # labels = batch['labels']
            outputs = self.model(**batch)

            if self.pretrained_mode:
                raise NotImplementedError('Unsupported pretrain!')
            
            # loss = self._sft_loss(outputs['logits'], labels)
            loss = outputs['loss']
            self.model.backward(loss)
            self.model.step()

            if self.args.print_loss:
                self.strategy.print(
                    f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
                )
            
            self._log_to_tb({'loss' : loss}, 'Train')
            if (step + 1) % int(len(self.train_dataloader) // 10) == 0:
                logs = self.evaluate()
                self.strategy.print(f'***** Evaluating model, Epoch {epoch + 1} / {self.args.num_train_epochs} *****')
                self.strategy.print(
                    f"perplexity : {logs['ppl']}, loss : {logs['loss']}"
                )
                self._log_to_tb(logs, 'Test')
                self.model.train()
                self.strategy.save_model(self.model.base_model, self.tokenizer)
        
    def train(self):
        self.strategy.print('***** Runing training *****')
        logs = self.evaluate()
        self.strategy.print(f'***** Evaluating model, Epoch {1} / {self.args.num_train_epochs} *****')
        self.strategy.print(
            f"perplexity : {logs['ppl']}, loss : {logs['loss']}"
        )
        self._log_to_tb(logs, 'Test')
        self.model.train()
        for epoch in range(self.args.num_train_epochs):
            self.strategy.print(
                f"Beginning of Epoch {epoch + 1}/{self.args.num_train_epochs}, Total Micro Batches {len(self.train_dataloader)}"
            )
            self.model.train()
            self._train_epoch(epoch)
            self.model.tput_timer.update_epoch_count()
        
        self.strategy.save_model(self.model.base_model, self.tokenizer)
    
