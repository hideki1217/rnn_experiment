from pathlib import Path
import re
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import parse
from PIL import Image

import sys
name = sys.argv[1]

cwd = Path(__file__).absolute().parent.parent / name
savedir = cwd / "result" / "trajectory" 
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
        plt.close(fig)

# make gif
for folder in filter(lambda x: x.is_dir(), savedir.iterdir()):
    pictures=[]
    for path in filter(lambda x: re.match(R"[0-9]*\.png", x.name), folder.iterdir()):
        (time, )=parse.parse("{:d}.png", path.name)
        img = Image.open(path)
        pictures.append((time, img))
    pictures = list(map(lambda x: x[1], sorted(pictures)))
    pictures[0].save(folder.parent / f'{folder.name}.gif',
                    save_all=True, 
                    append_images=pictures[1:], 
                    optimize=True, 
                    duration=500, 
                    loop=0)
        