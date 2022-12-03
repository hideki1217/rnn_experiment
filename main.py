from pathlib import Path
import subprocess
import sys


target = "./rnn_classify"
cp = subprocess.run(target, cwd=Path(__file__).parent / "build" / "src")
if cp.returncode != 0:
    print(f'{target}: failed.', file=sys.stderr)
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