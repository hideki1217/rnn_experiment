import manager
import utils

import torch
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import itertools


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = Path(__file__).parent / "time_evol"
    if not base.exists():
        base.mkdir()

    def naming(p: manager.Param):
        return f"(g_radius, batch, t_rnn, model_seed) = ({p.g_radius}, {p.num_batch}, {p.t_rnn}, {p.model_seed})"

    def do(p: manager.Param):
        utils.torch_fix_seed()
        model_key = hash(p)
        savepath = base / f"{model_key}.csv"
        if savepath.exists():
            return

        trainer = manager.Trainer(p, device)
        delta = []
        for e in range(p.num_epoch):
            w = trainer.model.w.data.clone()
            b = trainer.model.b.data.clone()
            w_out = trainer.model.w_out.clone()
            b_out = trainer.model.b_out.clone()

            loss = trainer.train_1epoch()

            delta_w = torch.norm(trainer.model.w.data - w).cpu().item()
            delta_b = torch.norm(trainer.model.b.data - b).cpu().item()
            delta_w_out = torch.norm(
                trainer.model.w_out.data - w_out).cpu().item()
            delta_b_out = torch.norm(
                trainer.model.b_out.data - b_out).cpu().item()

            delta.append((e, delta_w, delta_b, delta_w_out, delta_b_out, loss))

        df = pd.DataFrame(delta, columns=[
            "epoch", "delta_w", "delta_b", "delta_w_out", "delta_b_out", "loss"]).set_index("epoch")
        df.to_csv(base / f"{model_key}.csv")

        df.plot()
        plt.title(naming(p))
        plt.xlabel("epoch")
        plt.ylabel("delta or loss")
        plt.savefig(base / f"{model_key}.png")
        plt.close()

    num_seed = 10
    seeds = utils.make_seeds(num_seed)
    params = [
        {"g_radius": g_radius, "num_batch": num_batch, "t_rnn": t_rnn}
        for g_radius in [20, 250]
        for num_batch in [10, 60]
        for t_rnn in [10, 5, 1]
    ]

    for param in params:
        for seed in seeds:
            p = manager.Param(seed, **param)
            print(p)
            do(p)


if __name__ == "__main__":
    main()
