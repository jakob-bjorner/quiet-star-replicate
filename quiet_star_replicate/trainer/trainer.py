from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pprint
import numpy as np
from math import ceil
from quiet_star_replicate.model.model import get_grad_norm, get_model_param_norm
from quiet_star_replicate.data.data import get_shakespeare_collate_fn, get_train_val_test_datasets
import os
import logging
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, tokenizer, device, get_train_loss_fn, get_eval_loss_fn, train_dl, val_dl, epochs, eval_every, save_directory, experiment_logger, grad_acc):
        self.model: torch.nn.Module = model
        self.tokenizer = tokenizer
        self.device = device
        self.get_train_loss_fn = get_train_loss_fn
        self.get_eval_loss_fn = get_eval_loss_fn
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.epochs = epochs
        self.eval_every = eval_every # =100
        self.save_directory = save_directory
        self.experiment_logger = experiment_logger
        self.grad_acc = grad_acc

    def train(self):
        # train_losses = []
        # eval_losses = []
        log_dict = dict()
        base_lr = 0.001
        optim = torch.optim.AdamW(self.model.parameters(), lr=base_lr)
        i = 0
        d = None
        logger.info(f'Number training steps total: {len(self.train_dl) * self.epochs}')
        for epoch_i in range(self.epochs):
            for d in self.train_dl:
                if i % self.eval_every == 0:
                    eval_loss = self.eval()
                    log_dict["eval_loss"] = eval_loss
                d = d.to(self.device)
                optim.zero_grad()
                recoreded_loss = 0
                numels = 0
                train_loss_dict = dict()
                sub_ds = d.split(ceil(d.size(0) // self.grad_acc))
                for sub_di, sub_d in enumerate(sub_ds):
                    train_loss_dict = self.get_train_loss_fn(self.model, sub_d, print_stuff=(sub_di==len(sub_ds)-1)) 
                    loss = train_loss_dict.pop('loss') # the only thing which should need a gradient is loss, the rest should be fine.
                    loss.backward()
                    recoreded_loss += loss.item()
                    numels += train_loss_dict.pop('numel')
                    # I can determine the step size after I get the numels total in what I just back propped. Then it is easy?
                recoreded_loss = recoreded_loss / numels
                
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad /= numels
                log_dict.update({"loss": recoreded_loss, "grad_norm": get_grad_norm(self.model).item() / numels, "model_param_norm": get_model_param_norm(self.model).item()})
                log_dict.update(train_loss_dict)
                log_dict['epoch'] = epoch_i
                log_dict['step'] = i
                self.experiment_logger.log(log_dict, step=i)
                logger.info(pprint.pformat(log_dict))
                log_dict = dict()
                
                optim.step()
                i += 1
        eval_loss = self.eval()
        log_dict['eval_loss'] = eval_loss
        log_dict['epoch'] = self.epochs
        log_dict['step'] = i
        self.experiment_logger.log(log_dict, step=i)
        logger.info(pprint.pformat(log_dict))
        # logger.info(f'eval loss {eval_loss}')
        # eval_losses.append((i, eval_loss))
        torch.save(self.model, os.path.join(self.save_directory, "model.pt"))
        # plt.plot(*zip(*train_losses), label='train')
        # plt.plot(*zip(*eval_losses), label='eval')
        # plt.savefig(os.path.join(self.save_directory, "train_eval.png"))

        if isinstance(d, torch.Tensor) and hasattr(self.model, "sample"): # doesn't work for reward model, but nice to have for the language models
            d = d.to(self.device)
            logger.info(pprint.pformat(self.tokenizer.batch_detokenize(d[:, :100])))
            logger.info(pprint.pformat(self.tokenizer.batch_detokenize(self.model.sample(d[:, :100], max_gen_length=20))))

            logger.info(self.tokenizer.batch_detokenize(self.model(d).argmax(dim=-1)))
    def eval(self):
        losses = []
        numels = []
        with torch.no_grad():
            self.model.eval()
            for d in self.val_dl:
                d = d.to(self.device)
                for sub_d in d.split(ceil(d.size(0) / self.grad_acc)):
                    eval_loss_dict = self.get_eval_loss_fn(self.model, sub_d)
                    loss = eval_loss_dict['loss']
                    numel = eval_loss_dict['numel']
                    losses.append(loss)
                    numels.append(numel)
            self.model.train()
        return float(sum(losses)) / sum(numels)

class LatentRationaleTrainer(Trainer):
    def __init__(self,):
        ...

class VariableLengthLatentRationaleTrainer(LatentRationaleTrainer):
    def __init__(self,):
        ...

        