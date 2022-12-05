from pathlib import Path
import subprocess
from multiprocessing import Pool
import sys
import itertools


process_n = 4
target = "./exp0"
g_radius = [1, 20, 100, 250]
inner_dim = [2, 200]
patience = [1, 2, 3, 5]
def f(param):
    param = tuple(map(str, param))
    cp = subprocess.run([target, *param], cwd=Path(__file__).parent / "build" / "src")
    if cp.returncode != 0:
        print(f'{target}({param}): failed.', file=sys.stderr)
    return cp.returncode

with Pool(process_n) as p:
    res = p.map(f, itertools.product(g_radius, inner_dim, patience))

    if all(map(lambda x: x == 0, res)):
        print("success!!")
    else:
        print("some task failed")
        sys.exit(1)

scripts = [
    ["./script/trajectory.py", "./script/ED.py", "./script/learning_log.py", "./script/spectoram.py"],
]

for script_group in scripts:
    procs = []
    for script in script_group:
        proc = subprocess.Popen(["python3", script], cwd=Path(__file__).parent)
        procs.append(proc)

    for proc in procs:
        proc.wait()
    procs.clear()