import math
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import datasets
import models
import utils


@dataclass(frozen=True, eq=True)
class Param:
    model_seed: int

    dim_x: int = 2
    dim_h: int = 200
    dim_y: int = 2
    scale_w_out: float = 0.3
    g_radius: float = 20
    dt: float = 0.01
    t_rnn: int = 10

    opt_lr: float = 0.001
    opt_beta: float = 0.99
    patience: int = 5
    factor: float = 0.5
    num_epoch: int = 120
    num_iteration: int = 80
    num_batch: int = 10

    num_mixture: int = 60
    noise_scale: float = 0.02
    len_cube: float = 4

    num_test_iteration: int = num_mixture
    num_test_batch: int = 50

def show_model(p: Param):
    file = Path(__file__).parent / "models" / f"{hash(p)}.pth"
    if file.exists():
        model = models.RNN(p.dim_x, p.dim_h, p.dim_y, p.t_rnn)
        res = torch.load(file)
        res.pop("model_state_dict")
        print(res)
    else:
        print("Not found")

def search_model(p: Param):
    file = Path(__file__).parent / "models" / f"{hash(p)}.pth"
    if file.exists():
        model = models.RNN(p.dim_x, p.dim_h, p.dim_y, p.t_rnn)
        res = torch.load(file)
        model.load_state_dict(res["model_state_dict"])
        return model
    return None

class Trainer:
    def __init__(self, p: Param, device):
        self.device = device
        self.p = p
        self.model = self._init_model(p).to(device)
        self.train_dataloader, self.test_dataloader = self._init_dataloader(p)
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), p.opt_lr, p.opt_beta)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=p.factor, patience=p.patience, threshold=1e-7)
        self.criteria = torch.nn.CrossEntropyLoss()

    def train_1epoch(self):
        self.model.train()

        loss_sum = 0.0
        for X, Y in self.train_dataloader:
            X = X.to(self.device)
            Y = Y.to(self.device)

            self.optimizer.zero_grad()

            y = self.model(X)
            loss = self.criteria(y, Y)
            loss_sum += loss

            loss.backward()
            self.optimizer.step()

        loss_mean = loss_sum.cpu().item() / len(self.train_dataloader)
        self.scheduler.step(loss_mean)
        return loss_mean
    
    @torch.no_grad()
    def test(self):
        self.model.eval()

        ok = 0
        for X, Y in self.test_dataloader:
            X = X.to(self.device)
            Y = Y.to(self.device)

            y = self.model(X)
            ok += torch.sum(torch.argmax(y, dim=1) == Y).item()

        n = self.p.num_test_iteration * self.p.num_test_batch
        return ok / n

    
    def _init_model(self, p: Param):
        model = models.RNN(p.dim_x, p.dim_h, p.dim_y, p.t_rnn)
        with utils.rng_scope(157, self.device):
            # model.w_in.data = nn.init.orthogonal_(model.w_in.T).T
            model.w_in.data = torch.eye(p.dim_x, p.dim_h)
            model.w_out.data = nn.init.normal_(
                model.w_out, 0, p.scale_w_out / math.sqrt(p.dim_h))
            model.b_out.data = nn.init.zeros_(model.b_out.data)
            model.b.data = nn.init.zeros_(model.b)

        with utils.rng_scope(p.model_seed, self.device):
            J = nn.init.normal_(model.w.data, 0, 1.0 / math.sqrt(p.dim_h))
            model.w.data = J * (p.dt * p.g_radius) + torch.eye(p.dim_h) * (1 - p.dt)
        
        return model
    
    def _init_dataloader(self, p: Param):
        stream = datasets.GaussianMix(datasets.points_with_minimal_dist(p.dim_x, p.num_mixture, p.len_cube, p.noise_scale * 10),
                                      p.noise_scale)

        class Labeling:
            def __init__(self, labels) -> None:
                self.labels = labels

            def __call__(self, index):
                label = self.labels[index]
                return label

        labeling = Labeling(torch.tensor(
            list(range(stream.num_class))) % p.dim_y)

        test_idxs = torch.tensor(
            list(range(p.num_test_iteration * p.num_test_batch))) % stream.num_class
        test_dataloader = DataLoader(TensorDataset(stream.gen_manual(test_idxs), labeling(test_idxs)),
                                     batch_size=p.num_test_batch)

        train_dataloader = DataLoader(datasets.StreamDataset(stream, p.num_iteration * p.num_batch, y_transform=lambda x: labeling(x).item()),
                                      batch_size=p.num_batch)
        return train_dataloader, test_dataloader

def create_model(p: Param, device):
    res = search_model(p)
    if res is not None:
        return res

    trainer = Trainer(p, device)

    losses = [trainer.train_1epoch() for _ in range(p.num_epoch)]
    test_acc = trainer.test()

    print(f"{losses[0]} => {losses[-1]}")

    torch.save({
        "test_acc": test_acc,
        "train_last_lr": trainer.optimizer.param_groups[0]["lr"],
        "train_last_loss": losses[-1],
        "param": p,
        "model_state_dict": trainer.model.state_dict(),
    }, Path(__file__).parent / "models" / f"{hash(p)}.pth")

    return trainer.model
