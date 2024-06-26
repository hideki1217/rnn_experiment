import os

os.environ["OPENBLAS_NUM_THREADS"] = "8"

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.cm as mpl
import matplotlib.pyplot as plt
import parse

import sys
name = sys.argv[1]

num_top = 10
num_all = 200

cwd = Path(__file__).absolute().parent.parent / name
savedir_all = cwd / "result" / "spectoram" 
if not savedir_all.exists():
    savedir_all.mkdir()
savedir = cwd / "result" / f"spectoram_{num_top}" 
if not savedir.exists():
    savedir.mkdir()

datadir = cwd / "log"
paths = list(filter(lambda x: x.exists(), map(lambda x: x / "spectoram.csv", datadir.iterdir())))
def _parse_param(file):
    beta, inner_dim, patience, _ = parse.parse(R"{:d}_{:d}_{:d}_{:d}", file.name)
    return beta, inner_dim, patience

def _param_stamp(param):
    param_stamp = f"{param[0]}_{param[1]}_{param[2]}"
    return param_stamp

param_dict = defaultdict(list)
for param, path in map(lambda x: (_parse_param(x.parent), x), paths):
    param_dict[param].append(path)

for param, ls in param_dict.items():
    stamp = _param_stamp(param)

    epoch_dst = defaultdict(list)
    for df in map(lambda x: pd.read_csv(x, index_col=0), ls):
        for epoch in df.index:
            epoch_dst[epoch].append(df.loc[epoch].values)
    epochs = sorted(list(epoch_dst.keys()))
    num_epoch = len(epochs)
    
    cmap = mpl.get_cmap("cool")
    colors = {name: cmap((i + 1) / num_epoch ) for i, name in enumerate(epochs)}
    
    fig, ax = plt.subplots()
    x = list(range(1, num_all + 1))
    for epoch in epochs:
        y_ls = np.array(epoch_dst[epoch])
        c = colors[epoch]

        mean = y_ls.mean(axis=0)

        for i in range(y_ls.shape[0]):
            ax.plot(x, y_ls[i], alpha=0.05, c=c)
        ax.plot(x, mean, c=colors[epoch], label=str(epoch))
    plt.xlabel(f"index")
    plt.ylabel(f"lyapunov exponent")
    plt.suptitle(stamp)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savedir_all/ f"{stamp}.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    x = list(range(1, num_top + 1))
    for epoch in epochs:
        y_ls = np.array(epoch_dst[epoch])
        c = colors[epoch]

        mean = y_ls.mean(axis=0)

        for i in range(y_ls.shape[0]):
            ax.plot(x, y_ls[i][:num_top], alpha=0.05, c=c)
        ax.plot(x, mean[:num_top], c=colors[epoch], label=str(epoch))
    plt.xlabel(f"index")
    plt.ylabel(f"lyapunov exponent")
    plt.suptitle(stamp)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savedir/ f"{stamp}.png")
    plt.close(fig)
