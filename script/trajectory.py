from pathlib import Path
import re
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import parse

savedir = Path(R"tmp/trajectory")
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

    save = savedir / param_stamp
    if not save.exists():
        save.mkdir()

    for t in range(1, span + 1):
        ncol = 4
        nrow = (n + ncol - 1) // ncol
        
        fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 3*nrow), tight_layout=True)
        fig.suptitle(f"t = {t}, {param_stamp}")

        for i, (seed, data, labels) in enumerate(datas):
            ax = axes[i//ncol, i%ncol]

            _, state = next(filter(lambda x: x[0] == t, data))
            label_set = set(labels)

            print(f"{param_stamp}:{t}: {seed}")

            pca = PCA(n_components=2)
            pcaed = pca.fit_transform(state)
            for label in label_set:
                index = labels == label
                pca1 = pcaed[index, 0]
                pca2 = pcaed[index, 1]
                ax.scatter(pca1, pca2)
            ax.set_xlabel("pca1")
            ax.set_ylabel("pca2")
        
        fig.savefig(save/f"{t}.png")
        plt.close()
        