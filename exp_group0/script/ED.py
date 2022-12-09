from pathlib import Path
import re
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import parse

import sys
name = sys.argv[1]

cwd = Path(__file__).absolute().parent.parent / name
savedir = cwd / "result" / "ED" 
if not savedir.exists():
    savedir.mkdir()

datadir = cwd / "log"
param_dict = defaultdict(list)
for data in datadir.glob(R"*_*_*_*"):
    w_beta, inner_dim, patience, seed = parse.parse(R"{:d}_{:d}_{:d}_{:d}", data.name)
    def _f(path):
        (time,) = parse.parse(R"{:d}.csv", path.name) 
        return time, path

    labels = pd.read_csv(data/"labels.csv", header=None).values.reshape((-1,))
    data = [_f(csv) for csv in filter(lambda x: re.match(R"[0-9]*\.csv", x.name), data.iterdir())]
    data.sort()
    param_dict[(w_beta, inner_dim, patience)].append((seed, data, labels))

for param, datas in param_dict.items():
    param_stamp = f"{param[0]}_{param[1]}_{param[2]}"

    datas = [(seed, 
              [(time, pd.read_csv(path, header=None)) for time, path in data], 
              labels) for seed, data, labels in datas]

    n = len(datas)
    span = len(datas[0][1])
    
    fig, ax = plt.subplots()
    fig.suptitle(f"{param_stamp}")

    x = None
    EDss = []
    for i, (seed, data, labels) in enumerate(datas):
        EDs = []
        label_set = set(labels)
        for t, state in data:
            cov = np.cov(state.T)
            eigvals = np.linalg.eigvals(cov)
            ED = sum(eigvals)**2 / sum(eigvals * eigvals)
            EDs.append((t, ED))
        
        EDs = np.array(sorted(EDs)).T
        if x is None:
            x = EDs[0]
        else:
            assert np.array_equal(x, EDs[0])
        EDss.append(EDs[1])
    EDss = np.array(EDss)

    for i in range(EDss.shape[0]):
        ax.plot(x, EDss[i], alpha=0.5)
    ax.plot(x, EDss.mean(axis=0))

    plt.xlabel("time")
    plt.ylabel("ED(t)")
    plt.savefig(savedir/f"{param_stamp}.png")