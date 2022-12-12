from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import parse

import sys
name = sys.argv[1]

cwd = Path(__file__).absolute().parent.parent / name
savedir = cwd / "result" / "learning_log" 
if not savedir.exists():
    savedir.mkdir()

datadir = cwd / "log"
paths = list(filter(lambda x: x.exists(), map(lambda x: x / "learning_log.csv",  datadir.iterdir())))
def _parse_param(file):
    beta, inner_dim, patience, _ = parse.parse(R"{:d}_{:d}_{:d}_{:d}", file.name)
    return beta, inner_dim, patience

param_dict = defaultdict(list)
for param, path in map(lambda x: (_parse_param(x.parent), x), paths):
    param_dict[param].append(path)

for param in param_dict:
    all = param_dict[param]
    all_df = [pd.read_csv(x, index_col=0) for x in all]
    param_stamp = f"{param[0]}_{param[1]}_{param[2]}"

    n = len(all_df[0].columns)
    fig, axes = plt.subplots(n, 1, sharex="col", figsize=(6, 3*n))
    for df in all_df:
        for i, s in enumerate(df.columns):
            axes[i].plot(df.index, df[s], alpha=0.5)
    for i, s in enumerate(all_df[0].columns):
        x = None
        for df in all_df:
            x = df[s] if x is None else x + df[s]
        x /= len(all_df)
        axes[i].plot(x.index, x.values)
        
        axes[i].set_ylabel(s)
    plt.xlabel("No. of training samples")
    plt.tight_layout()
    plt.savefig(savedir/f"{param_stamp}.png")
    plt.close(fig)
