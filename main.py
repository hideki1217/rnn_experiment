import subprocess

interp = r"/home/okumura/workspace/rnn_experiment/venv/bin/python"
subprocess.run([interp, "exp0/run.py"])
print("exp0/run.py: fin")
subprocess.run([interp, "exp1/run.py"])
print("exp1/run.py: fin")
