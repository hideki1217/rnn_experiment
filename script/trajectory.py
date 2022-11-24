from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import parse

workspace = Path(R"tmp/trajectory")
if not workspace.exists():
    workspace.mkdir()

savedir = Path(R"tmp/trajectory")
if not savedir.exists():
    savedir.mkdir()

datadir = Path(R"log")
for data in datadir.glob(R"*_*_*_*"):
    w_beta, inner_dim, patience, seed = parse.parse(R"{:d}_{:d}_{:d}_{:d}", data.name)

    save = savedir / data.name
    if not save.exists():
        save.mkdir()

    labels = pd.read_csv(data/"labels.csv", header=None).values.reshape((-1,))
    label_set = set(labels)
    for csv in filter(lambda x: re.match(R"[0-9]*\.csv", x.name), data.iterdir()):
        (time,) = parse.parse(R"{:d}.csv", csv.name) 
        state = pd.read_csv(csv, header=None)

        print(state.head())

        scalar = StandardScaler(with_std=False)
        state = scalar.fit_transform(state)

        pca = PCA(n_components=state.shape[1])
        pcaed = pca.fit_transform(state)

        fig, ax = plt.subplots()
        for label in label_set:
            index = labels == label
            pca1 = pcaed[index, 0]
            pca2 = pcaed[index, 1]
            ax.scatter(pca1, pca2)
        ax.set_xlabel("pca1")
        ax.set_ylabel("pca2")
        ax.set_title(f"t = {time}, {w_beta}, {inner_dim}, {patience}, {seed}")
        fig.savefig(save/f"{time}.png")



