import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("log/rnn_classify.csv", index_col=0, header=None)
df.rename(columns={1: "batch_score", 2: "test_score",
          3: "test_acc"}, inplace=True)

df.plot(subplots=True)
plt.savefig("tmp/rnn_dim2_beta20.png")
