from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import parse

savedir = Path(R"tmp/spectoram")
if not savedir.exists():
    savedir.mkdir()

datadir = Path(R"log")
param_dict = defaultdict(list)
for data in datadir.glob(R"*_*_*_*"):
    pass
