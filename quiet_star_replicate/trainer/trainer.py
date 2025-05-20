from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pprint
import numpy as np
from math import ceil
from quiet_star_replicate.model.model import get_grad_norm, get_model_param_norm
from quiet_star_replicate.data.data import get_shakespeare_collate_fn, get_train_val_test_datasets
from accelerate.utils.memory import clear_device_cache
import os
import logging
logger = logging.getLogger(__name__)
from functools import partial


def linear_warm_fun(step, max_steps, frac_warm=0.1):
    warmup = int(max_steps * frac_warm)
    if step < warmup:
        return step / warmup
    else: 
        new_step = step - warmup
        new_max_steps = (max_steps-warmup)
        if new_max_steps == 0:
            return 1
        return 1 - new_step / new_max_steps
class Trainer:
    def __init__(self, model, tokenizer, device, get_train_loss_fn, get_eval_loss_fn, train_dl, val_dl, base_lr, epochs, max_steps, eval_every, save_directory, experiment_logger, grad_acc, use_scheduler, schedule_punfinished, punish_unfinished, clip_gradients, neuter_dropout_base):
        self.model: torch.nn.Module = model
        self.tokenizer = tokenizer
        self.device = device
        self.get_train_loss_fn = get_train_loss_fn
        self.get_eval_loss_fn = get_eval_loss_fn
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.base_lr = base_lr
        self.epochs = epochs
        self.max_steps = max_steps
        self.eval_every = eval_every # = 100
        self.save_directory = save_directory
        self.experiment_logger = experiment_logger
        self.grad_acc = grad_acc
        self.use_scheduler = use_scheduler
        self.schedule_punfinished = schedule_punfinished
        self.punish_unfinished = punish_unfinished
        self.clip_gradients = clip_gradients
        self.neuter_dropout_base = neuter_dropout_base

    def test_cuda_mem_limit(self):
        '''tests the upper limits of cuda memory, so no OOM will be hit during training.'''
        from copy import deepcopy
        model_temp = deepcopy(self.model)
        optim_temp = torch.optim.AdamW(self.model.parameters(), lr=0.0)

        for d in self.train_dl:
            d = d.to(self.device)
            optim_temp.zero_grad()
            numels = 0
            sub_ds = d.split(ceil(d.size(0) // self.grad_acc))
            for sub_di, sub_d in enumerate(sub_ds):
                train_loss_dict = self.get_train_loss_fn(model_temp, sub_d, print_stuff=(sub_di==len(sub_ds)-1), test_mem=True) 
                loss = train_loss_dict.pop('loss') # the only thing which should need a gradient is loss, the rest should be fine.
                loss.backward()
                numels += train_loss_dict.pop('numel')
                # I can determine the step size after I get the numels total in what I just back propped. Then it is easy?
            
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad /= numels
            optim_temp.step()
            break

    def train(self):
        self.test_cuda_mem_limit()
        clear_device_cache(garbage_collection=True)
        print("MEM check done, shouldn't OOM below this.")
        # train_losses = []
        # eval_losses = []
        log_dict = dict()
        base_lr = self.base_lr
        if hasattr(self.model, "get_param_groups"):
            reasoner_lr = base_lr * self.model.model.base_language_model.hidden_size / self.model.model.base_reasoner.hidden_size * 4
            groups = self.model.get_param_groups(reasoner_lr)
            optim = torch.optim.AdamW(groups, lr=base_lr)
        else:
            optim = torch.optim.AdamW(self.model.parameters(), lr=base_lr) 
        # elif self.parameter_groups == 1: # do different learning rate on the parameters which correspond to the reasoner size.
        #     optim = torch.optim.AdamW([
        #         {"params": self.model.lm_head.parameters()},
        #         {"params": self.model.model.base_language_model_token_embedding.parameters()},
        #         {"params": self.model.model.base_language_model.parameters()},
        #         {"params": self.model.model.base_reasoner_token_embedding.parameters(), "lr": reasoner_lr},
        #         {"params": self.model.model.base_reasoner.parameters(), "lr": reasoner_lr},
        #         {"params": self.model.model.reasoner.parameters(), "lr": reasoner_lr},
        #         {"params": self.model.model.interpreter_token_embedding.parameters(), "lr": reasoner_lr},
        #         {"params": self.model.model.interpreter.parameters(), "lr": reasoner_lr},
        #         {"params": self.model.model.downproject_interpreter_rep.parameters(), "lr": reasoner_lr},
        #         {"params": self.model.model.mixer_norm_interpeter.parameters(), "lr": reasoner_lr},
        #         {"params": self.model.model.mixer_norm_base_lm.parameters(), "lr": reasoner_lr},
        #         {"params": self.model.model.combining_base_lm_and_interpreter_residual.parameters(), "lr": reasoner_lr}], 
        #         lr=base_lr) 
        # else: raise NotImplementedError()
        i = 0
        d = None
        if self.max_steps == -1:
            max_steps = len(self.train_dl) * self.epochs
        elif self.max_steps == -2: # just a code to make the model run 50 epochs over the same subset so break in the epoch every 100 steps, and then restart. Need to ensure that the same data is at the start every time.
            max_steps = 2700
        else:
            max_steps = self.max_steps

        pun_schedule = partial(linear_warm_fun, max_steps=max_steps, frac_warm=1)
        
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, partial(linear_warm_fun, max_steps=max_steps) if self.use_scheduler else (lambda step: 1)) 
        logger.info(f'Number training steps total: {max_steps}')
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
                if self.schedule_punfinished:
                    self.get_train_loss_fn = partial(self.get_train_loss_fn, punish_unfinished=pun_schedule(i) * self.punish_unfinished)
                if self.neuter_dropout_base > 0:
                    # self.model.model.base_language_model_dropout.p = self.neuter_dropout_base * (i % 5)/4 # putting this here because this could require a schedule.
                    # self.model.model.dropout_index = (i % 5)
                    self.model.model.base_language_model_dropout.p = self.neuter_dropout_base # temp to check that the reasoning is actually being used.
                    self.model.model.dropout_index = 0
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
                log_dict.update({"loss": recoreded_loss, "grad_norm": get_grad_norm(self.model).item(), "model_param_norm": get_model_param_norm(self.model).item()})
                if self.clip_gradients != 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clip_gradients)
                log_dict.update(train_loss_dict)
                log_dict['epoch'] = epoch_i
                log_dict['step'] = i
                if self.schedule_punfinished:
                    log_dict['punfinished'] = pun_schedule(i) * self.punish_unfinished
                else:
                    log_dict['punfinished'] = self.punish_unfinished
                optim.step()
                if self.use_scheduler:
                    log_dict['lr'] = lr_scheduler.get_last_lr()[-1]
                    lr_scheduler.step()
                self.experiment_logger.log(log_dict, step=i)
                logger.info(pprint.pformat(log_dict))
                log_dict = dict()
                i += 1
                if i >= max_steps:
                    break
                if self.max_steps == -2:
                    if i % 100 == 0:
                        break # this is a special case for testing repeating elements in the dataset.
            if i >= max_steps:
                break
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
            self.model.eval()
            d = d.to(self.device)
            logger.info(pprint.pformat(self.tokenizer.batch_detokenize(d[:, :100])))
            # import ipdb; ipdb.set_trace()
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

        