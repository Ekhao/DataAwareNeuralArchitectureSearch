import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np

run1 = pd.read_csv(
    "experiment/results/pareto_front_exp1_run1.csv", index_col="Model Number")
run2 = pd.read_csv(
    "experiment/results/pareto_front_exp1_run2.csv", index_col="Model Number")
run3 = pd.read_csv(
    "experiment/results/pareto_front_exp1_run3.csv", index_col="Model Number")
run4 = pd.read_csv(
    "experiment/results/pareto_front_exp1_run4.csv", index_col="Model Number")
run5 = pd.read_csv(
    "experiment/results/pareto_front_exp1_run5.csv", index_col="Model Number")


# Calculate F1 score for each row
# Formula taken from: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html


def add_f1_score(row):
    return 2 * (row["Precision"] * row["Recall"]) / (row["Precision"] + row["Recall"])


run1["F1"] = run1.apply(lambda row: add_f1_score(row), axis=1)
run2["F1"] = run2.apply(lambda row: add_f1_score(row), axis=1)
run3["F1"] = run3.apply(lambda row: add_f1_score(row), axis=1)
run4["F1"] = run4.apply(lambda row: add_f1_score(row), axis=1)
run5["F1"] = run5.apply(lambda row: add_f1_score(row), axis=1)


# Plot F1 score against Model Size
plt.close("all")
_, ax = plt.subplots()
ax.plot(run1["Model Size"], run1["F1"], label="Run 1")
ax.plot(run2["Model Size"], run2["F1"], label="Run 2")
ax.plot(run3["Model Size"], run3["F1"], label="Run 3")
ax.plot(run4["Model Size"], run4["F1"], label="Run 4")
ax.plot(run5["Model Size"], run5["F1"], label="Run 5")
ax.set_xlim(16000, 8000)
# ax.grid()
#ax.set_ylim(0, 4)
ax.set_xlabel("Model Size in Bytes")
ax.set_ylabel("F1-Score")
ax.legend(loc="upper right")
plt.savefig(f"figures/pareto-normal.png", format="png")
