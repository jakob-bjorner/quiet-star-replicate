from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pprint import pprint
from quiet_star_replicate.model.model import get_grad_norm, get_model_param_norm
from quiet_star_replicate.data.data import get_shakespeare_collate_fn, get_train_val_test_datasets
class Trainer(ABC):
    def __init__(self, device, get_loss_fn, train_dl, eval_dl, epochs, eval_every):
        self.device = device
        self.eval_dl = eval_dl
        self.train_dl = train_dl
        self.epochs = epochs
        self.eval_every = eval_every # =100
        self.get_loss_fn = get_loss_fn

        # train_dl = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    def train(self, model, tokenizer, print_stuff=True):
        train_losses = []
        eval_losses = []


        optim = torch.optim.AdamW(model.parameters())
        i = 0
        d = None
        if print_stuff:
            print('Number training steps total:', len(self.train_dl) * self.epochs)
        for epoch_i in range(self.epochs):
            for d in self.train_dl:
                if i % self.eval_every == 0 and print_stuff:
                    eval_loss = self.eval(model)
                    print('eval loss', eval_loss)
                    eval_losses.append((i, eval_loss))
                optim.zero_grad()
                loss = self.get_loss_fn(model, d)
                loss.backward()
                if print_stuff:
                    print(f"loss {i:>5}: {loss.item():<8.4f} grad norm: {get_grad_norm(model):<15.4f} model param norm: {get_model_param_norm(model):<15.4f}")
                train_losses.append((i, loss.item()))
                optim.step()
                i += 1

        if print_stuff:
            eval_loss = self.eval(model)
            print('eval loss', eval_loss)
            eval_losses.append((i, eval_loss))
            plt.plot(*zip(*train_losses), label='train')
            plt.plot(*zip(*eval_losses), label='eval')
            plt.show()

            if isinstance(d, torch.Tensor) and hasattr(model, "sample"): # doesn't work for reward model, but nice to have for the language models
                d = d.to(self.device)
                pprint(tokenizer.batch_detokenize(d[:, :100]))
                pprint(tokenizer.batch_detokenize(model.sample(d[:, :100], 20)))

                print(tokenizer.batch_detokenize(model(d).argmax(dim=-1)))
    def eval(self, model):
        losses = []
        with torch.no_grad():
            for d in self.eval_dl:
                loss = self.get_loss_fn(model, d)
                losses.append(loss)
        return float(sum(losses)) / len(losses)

class LatentRationaleTrainer(Trainer):
    def __init__(self,):
        ...

class VariableLengthLatentRationaleTrainer(LatentRationaleTrainer):
    def __init__(self,):
        ...

        