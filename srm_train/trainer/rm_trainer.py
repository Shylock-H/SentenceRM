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
from model import RewardModel
from utils.strategy import Strategy
from typing import Dict, List

class RMTrainer(ABC):
    def __init__(
        self,
        model : RewardModel,
        strategy : Strategy,
        optimizer : Optimizer,
        train_dataloader : DataLoader,
        eval_dataloader : DataLoader,
        scheduler : LRScheduler,
        tokenizer : PreTrainedTokenizer,
        max_norm : float = 0.5,             
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
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct_predictions, total_predictions = 0, 0
        chosen_scores = 0
        reject_scores = 0
        losses = 0.0
        for step, batch in enumerate(self.eval_dataloader):
            batch = self.strategy.to_device(batch)
            outputs = self.model(**batch)

            chosen, reject, loss = itemgetter(
                'chosen_rewards', 'reject_rewards', 'loss'
            )(outputs)

            correct_predictions += (chosen > reject).sum()
            total_predictions += chosen.shape[0]
            chosen_scores += chosen.mean().float()
            reject_scores += reject.mean().float()
            losses += loss.mean().float()
        
        acc = correct_predictions / total_predictions
        chosen_scores = chosen_scores / (step + 1)
        reject_scores = reject_scores / (step + 1)
        losses = losses / (step + 1)
        try:
            acc = self.strategy.all_reduce(acc, 'mean').item()
            chosen_scores = self.strategy.all_reduce(chosen_scores, 'mean').item()
            reject_scores = self.strategy.all_reduce(reject_scores, 'mean').item()
            losses = self.strategy.all_reduce(losses, 'mean').item()
        except:
            pass

        return dict(
            loss = loss, 
            chosen_reward = chosen_scores, 
            reject_reward = reject_scores, 
            acc = acc
        )
    
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
    
    def _train_epoch(self, epoch : int):
        for step, batch in enumerate(self.train_dataloader):
            batch = self.strategy.to_device(batch)
            outputs = self.model(**batch, use_cache = False)
            loss = outputs['loss']
            self.model.backward(loss)
            self.model.step()
            
            if self.args.print_loss:
                self.strategy.print(
                    f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
                )
            
            self._log_to_tb(outputs, 'Train')
            
            if (step + 1) % int(len(self.train_dataloader) // 10) == 0:
                logs = self.evaluate()
                self.strategy.print(f'***** Evaluating reward model, Epoch {epoch + 1} / {self.args.num_train_epochs} *****')
                self.strategy.print(
                    f"chosen_scores : {logs['chosen_reward']}, accuracy : {logs['acc']}"
                )
                self._log_to_tb(logs, 'Test')
                self.model.train()
                self.strategy.save_model(self.model, self.tokenizer)
            
    def train(self):
        logs = self.evaluate()
        self.strategy.print(f'***** Evaluating reward model, Epoch {1} / {self.args.num_train_epochs} *****')
        self.strategy.print(
            f"chosen_scores : {logs['chosen_reward']}, accuracy : {logs['acc']}"
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
        
        self.strategy.save_model(self.model, self.tokenizer)
        