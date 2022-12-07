from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import parse


cwd = Path(__file__).parent.parent
savedir = cwd / "result" / "spectoram" 
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

    n = 5
    ncols = n
    nrows = 1
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3 * n, 6), sharey="row")
    xs = [None] * n
    for df in map(lambda x: pd.read_csv(x, index_col=0), ls):
        df.sort_index(inplace=True)
        for i in range(n):
            ax = axes[i]
            x = df.iloc[:, i]
            ax.plot(x.index, x.values, alpha=0.5)

            xs[i] = x if xs[i] is None else xs[i] + x
    
    for i in range(n):
        x = xs[i] / len(ls)
        axes[i].plot(x.index, x.values)

    for i in range(n):
        ax = axes[i]
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{i}th lyapunov exponent")
    plt.suptitle(stamp)
    plt.tight_layout()
    fig.savefig(savedir/ f"{stamp}.png")
    plt.clf()
