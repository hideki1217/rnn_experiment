from pathlib import Path
import re
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import parse

savedir = Path(R"tmp/ED")
if not savedir.exists():
    savedir.mkdir()

datadir = Path(R"log")
param_dict = defaultdict(list)
for data in datadir.glob(R"*_*_*_*"):
    w_beta, inner_dim, patience, seed = parse.parse(R"{:d}_{:d}_{:d}_{:d}", data.name)
    def _f(path):
        (time,) = parse.parse(R"{:d}.csv", path.name) 
        state = pd.read_csv(path, header=None)
        return time, state

    labels = pd.read_csv(data/"labels.csv", header=None).values.reshape((-1,))
    data = [_f(csv) for csv in filter(lambda x: re.match(R"[0-9]*\.csv", x.name), data.iterdir())]
    data.sort()
    param_dict[(w_beta, inner_dim, patience)].append((seed, data, labels))

for param, datas in param_dict.items():
    param_stamp = f"{param[0]}_{param[1]}_{param[2]}"

    n = len(datas)
    span = len(datas[0][1])
    
    fig, ax = plt.subplots()
    fig.suptitle(f"{param_stamp}")

    
    for i, (seed, data, labels) in enumerate(datas):
        EDs = []
        label_set = set(labels)
        for t, state in data:
            cov = np.cov(state.T)
            eigvals = np.linalg.eigvals(cov)
            ED = sum(eigvals)**2 / sum(eigvals * eigvals)
            EDs.append((t, ED))
        
        EDs = np.array(sorted(EDs)).T
        ax.plot(EDs[0], EDs[1])

    plt.xlabel("time")
    plt.ylabel("ED(t)")
    plt.savefig(savedir/f"{param_stamp}.png")