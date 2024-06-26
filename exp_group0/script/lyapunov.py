import os

os.environ["OPENBLAS_NUM_THREADS"] = "4"

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import parse
import multiprocessing

class RNN:
    def __init__(self, weight, bias) -> None:
        assert weight.shape[0] == weight.shape[1]
        assert weight.shape[1] == bias.shape[0]
        self.n = weight.shape[0]
        self._w = weight
        self._b = bias
    
    def __call__(self, h):
        return np.tanh(self._w @ h + self._b)
    
    def d(self, h):
        y = self(h)
        return np.diag(1 - y * y) @ self._w


def lyapunov_exponent(model, max_iter = 1000):
    LEs = np.zeros((model.n,))
    h = np.ones((model.n,))
    Q = np.eye(N=model.n)
    for _ in range(max_iter):
        h = model(h)
        Df = model.d(h)
        A = Df @ Q
        Q, R = np.linalg.qr(A)
        LEs += np.log(np.abs(np.diag(R)))
    
    LEs = LEs / max_iter
    return LEs

def calc_lyap(file):
    max_iter = 1000
    (epoch, ) = parse.parse(r"model_{:d}.csv", file.name)
    arr = np.loadtxt(file, dtype=np.float64, delimiter=",",)
    
    model = RNN(weight=arr[:-1], bias=arr[-1])
    LEs = lyapunov_exponent(model, max_iter=max_iter)

    return (epoch, LEs)

import sys
name = sys.argv[1]

cwd = Path(__file__).absolute().parent.parent / name

datadir = cwd / "log"
params = [data for data in datadir.glob(R"*_*_*_*")]
def f(data):
    dst = data / "spectoram.csv"
    if dst.exists():
        return
    
    models = list(data.glob("model_*.csv"))
    les = np.array([np.insert(np.sort(les)[::-1], 0, epoch) for epoch, les in map(calc_lyap , models)])
    np.savetxt(data/"spectoram.csv", les, delimiter=",")

# from threadpoolctl import threadpool_info
# from pprint import pp
# pp(threadpool_info())

# for param in params:
#     f(param)
with multiprocessing.Pool(8) as p:
    p.map(f, params)
