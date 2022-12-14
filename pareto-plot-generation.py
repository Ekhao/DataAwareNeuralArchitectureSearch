import matplotlib.pyplot as plt
import pandas as pd

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

# Sort dataframes by model size
run1 = run1.sort_values(by=["Model Size"])
run2 = run2.sort_values(by=["Model Size"])
run3 = run3.sort_values(by=["Model Size"])
run4 = run4.sort_values(by=["Model Size"])
run5 = run5.sort_values(by=["Model Size"])


# Plot F1 score against Model Size
plt.close("all")
_, ax = plt.subplots()
ax.plot(run1["Model Size"], run1["F1"], "-k", label="Run 1")
ax.plot(run2["Model Size"], run2["F1"], "--k", label="Run 2")
ax.plot(run3["Model Size"], run3["F1"], "-.k", label="Run 3")
ax.plot(run4["Model Size"], run4["F1"], ":k", label="Run 4")
ax.plot(run5["Model Size"], run5["F1"], ".-k", label="Run 5")
ax.set_xlim(16000, 7000)
ax.set_ylim(0.2, 1.2)
ax.grid()
ax.set_xlabel("Model Size in Bytes")
ax.set_ylabel("F1-Score")
ax.legend(loc="lower left")

for i, label in enumerate(run1.index.to_numpy()):
    plt.annotate(
        label, (run1.iloc[i]["Model Size"]+100, run1.iloc[i]["F1"]+0.03))
for i, label in enumerate(run2.index.to_numpy()):
    plt.annotate(label, (run2.iloc[i]["Model Size"], run2.iloc[i]["F1"]))
for i, label in enumerate(run3.index.to_numpy()):
    plt.annotate(label, (run3.iloc[i]["Model Size"], run3.iloc[i]["F1"]))
for i, label in enumerate(run4.index.to_numpy()):
    plt.annotate(label, (run4.iloc[i]["Model Size"], run4.iloc[i]["F1"]))
for i, label in enumerate(run5.index.to_numpy()):
    plt.annotate(label, (run5.iloc[i]["Model Size"], run5.iloc[i]["F1"]))


plt.savefig(f"figures/pareto-normal.png", format="png")
