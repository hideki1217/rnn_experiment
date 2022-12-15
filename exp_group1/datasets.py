import torch
from torch.utils.data import Dataset

class DataStream:
    def gen(self, n=1):
        pass


class StreamDataset(Dataset):
    def __init__(self, stream: DataStream, n, x_transform=None, y_transform=None) -> None:
        super().__init__()
        self._stream = stream
        self._n = n
        self._x_trans = x_transform
        self._y_trans = y_transform
    
    def __len__(self):
        return self._n
    
    def __getitem__(self, _):
        x, y = self._stream.gen(1)
        x = torch.flatten(x)
        y = y[0].item()
        if self._x_trans is not None:
            x = self._x_trans(x)
        if self._y_trans is not None:
            y = self._y_trans(y)
        return x, y


class GaussianMix(DataStream):
    def __init__(self, means, scale, seed=100):
        self._engine = torch.Generator().manual_seed(seed)
        self.means = means
        self.scale = scale
    
    @property
    def num_class(self):
        return self.means.shape[0]
    
    def gen_manual(self, idxs):
        means = self.means[idxs]
        noise = torch.randn(means.size(), generator=self._engine) * self.scale
        return means + noise

    def gen(self, n=1):
        idxs = torch.randint(0, self.num_class - 1, (n,), generator=self._engine)
        x = self.gen_manual(idxs)
        return x, idxs


def points_with_minimal_dist(dim, n, l, minimal_dist, seed=120):
    engine = torch.Generator().manual_seed(seed)
    ls = []
    threshold = minimal_dist ** 2
    for i in range(n):
        flag = True
        while flag:
            tmp = torch.rand(dim, generator=engine) * l - l/2

            flag = False
            for j in range(i):
                dist = torch.sum((ls[j] - tmp)**2).item()
                if dist < threshold:
                    flag = True
                    break
        ls.append(tmp)
    
    ls = torch.cat(ls).reshape(len(ls), *ls[0].shape)
    return ls
