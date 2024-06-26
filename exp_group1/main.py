from dataclasses import replace

import manager
import lyap
import utils

import torch
import time


def do(make_p):
    p: manager.Param = make_p(None)
    print(f"## {p}")

    t_sum = 0
    LEs_sum = torch.zeros(p.dim_h)
    for seed in seeds:
        utils.torch_fix_seed()
        p = make_p(seed)

        start = time.time()
        model = manager.create_model(p, device=device)
        t_sum += time.time() - start

        LEs = lyap.lyapunov_exponent(model, device=device).cpu()
        LEs, indices = torch.sort(LEs, descending=True)
        LEs_sum += LEs[indices]

    print(f"mean of create model time: {t_sum / len(seeds)}(s)")
    print(f"mean of LEs: {LEs_sum[:10] / len(seeds)}")


num_seed = 10
seeds = utils.make_seeds(num_seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

make_p_ls = [
    lambda x: manager.Param(x),
    lambda x: manager.Param(x, g_radius=250),
    lambda x: manager.Param(x, num_batch=60),
    lambda x: manager.Param(x, num_batch=60, g_radius=250),
]

for make_p in make_p_ls:
    do(make_p)
