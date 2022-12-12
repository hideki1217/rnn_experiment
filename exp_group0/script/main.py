from pathlib import Path
import subprocess
from multiprocessing import Pool
import itertools

import sys
name = sys.argv[1]

target = "./" + name

cwd=Path(__file__).absolute()

g_radius = [1, 20, 100, 250]
inner_dim = [2, 200]
patience = [1, 2, 3, 5]
def f(param):
    param = tuple(map(str, param))
    cp = subprocess.run([target, *param], cwd=cwd.parent.parent.parent / "build" / "exp_group0")
    if cp.returncode != 0:
        print(f'{target}({param}): failed.', file=sys.stderr)
    return cp.returncode

with Pool(8) as p:
    res = p.map(f, itertools.product(g_radius, inner_dim, patience))

    if all(map(lambda x: x == 0, res)):
        print("success!!")
    else:
        print("some task failed")
        sys.exit(1)