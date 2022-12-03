from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import parse

savedir = Path(R"tmp/spectoram")
if not savedir.exists():
    savedir.mkdir()

paths = list(filter(lambda x: x.exists(), map(lambda x: x / "spectoram.csv", Path("log").iterdir())))
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

    fig, ax = plt.subplots()
    for df in map(lambda x: pd.read_csv(x, index_col=0), ls):
        maxs = df.max(axis=1)
        ax.plot(maxs.index, maxs.values)
    ax.set_xlabel("epoch")
    ax.set_ylabel("max lyapunov exponent")
    ax.set_title(stamp)
    fig.savefig(savedir/ f"{stamp}.png")
