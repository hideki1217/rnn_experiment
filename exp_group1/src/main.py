import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import datasets
import models
import utils

def main():
    dim_x = 2
    dim_h = 200
    dim_y = 2
    scale_w_out = 0.3
    g_radius = 20
    dt = 0.01
    t_rnn = 10
    model_seed = 145

    opt_lr = 0.001
    opt_beta = 0.99
    patience = 5
    factor = 0.5
    num_epoch = 120
    num_iteration = 80
    num_batch = 60

    num_mixture = 60
    noise_scale = 0.02
    len_cube = 4

    num_test_iteration = num_mixture
    num_test_batch = 50
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def make_model(model_seed):
        model = models.RNN(dim_x, dim_h, dim_y, t_rnn)

        with utils.rng_scope(157):
            # model.w_in.data = nn.init.orthogonal_(model.w_in.T).T
            model.w_in.data = torch.eye(dim_x, dim_h)
            model.w_out.data = nn.init.normal_(model.w_out, 0, scale_w_out / math.sqrt(dim_h))
        
        with utils.rng_scope(model_seed):
            model.w.data = nn.init.normal_(model.w, 0, g_radius / math.sqrt(dim_h)) * dt + torch.eye(dim_h) * (1 - dt)
            model.b.data = nn.init.zeros_(model.b)

        return model
    
    def make_dataloader():
        stream = datasets.GaussianMix(datasets.points_with_minimal_dist(dim_x, num_mixture, len_cube, noise_scale * 10), 
                                noise_scale)

        class Labeling:
            def __init__(self, labels) -> None:
                self.labels = labels
            
            def __call__(self, index):
                label = self.labels[index]
                return label

        labeling = Labeling(torch.tensor(list(range(stream.num_class))) % dim_y)

        test_idxs = torch.tensor(list(range(num_test_iteration * num_test_batch))) % stream.num_class
        test_dataloader = DataLoader(TensorDataset(stream.gen_manual(test_idxs), labeling(test_idxs)), batch_size=num_test_batch)
        
        train_dataloader = DataLoader(datasets.StreamDataset(stream, num_iteration * num_batch, y_transform=lambda x: labeling(x).item()), batch_size=num_batch)
        return train_dataloader, test_dataloader

    model = make_model(model_seed).to(device)    
    train_dataloader, test_dataloader = make_dataloader()

    def train():
        model.train()

        optimizer = torch.optim.RMSprop(model.parameters(), opt_lr, opt_beta)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience, threshold=1e-7)
        criteria = torch.nn.CrossEntropyLoss()

        for e in range(num_epoch):
            loss_sum = 0.0
            for X, Y in train_dataloader:
                X = X.to(device)
                Y = Y.to(device)

                optimizer.zero_grad()

                y = model(X)
                loss = criteria(y, Y)
                loss_sum += loss

                loss.backward()
                optimizer.step()

            scheduler.step(loss_sum.item() / num_iteration)
            print(f"{e}: batch_loss = {loss_sum.item() / num_iteration}")
    
    @torch.no_grad()
    def test():
        ok = 0
        for X, Y in test_dataloader:
            X = X.to(device)
            Y = Y.to(device)

            y = model(X)
            ok += torch.sum(torch.argmax(y, dim=1) == Y).item()
        
        n = num_test_iteration * num_test_batch
        print(f"test accuracy = {ok / n}({ok}/{n})")

    train()
    
    model.eval()
    test()


if __name__ == "__main__":
    main()
