import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

font = {"size": 12}

run11 = pd.read_csv(
    "../original_experiment/results/pareto_front_exp1_run1.csv", index_col="Model Number")
run12 = pd.read_csv(
    "../original_experiment/results/pareto_front_exp1_run2.csv", index_col="Model Number")
run13 = pd.read_csv(
    "../original_experiment/results/pareto_front_exp1_run3.csv", index_col="Model Number")
run14 = pd.read_csv(
    "../original_experiment/results/pareto_front_exp1_run4.csv", index_col="Model Number")
run15 = pd.read_csv(
    "../original_experiment/results/pareto_front_exp1_run5.csv", index_col="Model Number")

run21 = pd.read_csv(
    "../original_experiment/results/pareto_front_exp2_run1.csv", index_col="Model Number")
run22 = pd.read_csv(
    "../original_experiment/results/pareto_front_exp2_run2.csv", index_col="Model Number")
run23 = pd.read_csv(
    "../original_experiment/results/pareto_front_exp2_run3.csv", index_col="Model Number")
run24 = pd.read_csv(
    "../original_experiment/results/pareto_front_exp2_run4.csv", index_col="Model Number")
run25 = pd.read_csv(
    "../original_experiment/results/pareto_front_exp2_run5.csv", index_col="Model Number")

# Calculate F1 score for each row
# Formula taken from: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html


def add_f1_score(row):
    if row["Precision"] == 0 or row["Recall"] == 0:
        return 0
    return 2 * (row["Precision"] * row["Recall"]) / (row["Precision"] + row["Recall"])


run11["F1"] = run11.apply(lambda row: add_f1_score(row), axis=1)
run12["F1"] = run12.apply(lambda row: add_f1_score(row), axis=1)
run13["F1"] = run13.apply(lambda row: add_f1_score(row), axis=1)
run14["F1"] = run14.apply(lambda row: add_f1_score(row), axis=1)
run15["F1"] = run15.apply(lambda row: add_f1_score(row), axis=1)

run21["F1"] = run21.apply(lambda row: add_f1_score(row), axis=1)
run22["F1"] = run22.apply(lambda row: add_f1_score(row), axis=1)
run23["F1"] = run23.apply(lambda row: add_f1_score(row), axis=1)
run24["F1"] = run24.apply(lambda row: add_f1_score(row), axis=1)
run25["F1"] = run25.apply(lambda row: add_f1_score(row), axis=1)

# Sort dataframes by model size
run11 = run11.sort_values(by=["Model Size"])
run12 = run12.sort_values(by=["Model Size"])
run13 = run13.sort_values(by=["Model Size"])
run14 = run14.sort_values(by=["Model Size"])
run15 = run15.sort_values(by=["Model Size"])

run21 = run21.sort_values(by=["Model Size"])
run22 = run22.sort_values(by=["Model Size"])
run23 = run23.sort_values(by=["Model Size"])
run24 = run24.sort_values(by=["Model Size"])
run25 = run25.sort_values(by=["Model Size"])

# Plot F1 score against Model Size


def thousands(x, pos):
    'The two args are the value and tick position'
    return '%dK' % (x*1e-3)


mkformatter = FuncFormatter(thousands)

plt.close("all")
_, ax = plt.subplots()

ax.plot(run11["Model Size"], run11["F1"], "xk", label="Run 1")
ax.plot(run12["Model Size"], run12["F1"], "+k", label="Run 2")
ax.plot(run13["Model Size"], run13["F1"], "ok", label="Run 3")
ax.plot(run14["Model Size"], run14["F1"], "vk", label="Run 4")
ax.plot(run15["Model Size"], run15["F1"], "sk", label="Run 5")

ax.set_xlim(16000, 7000)

ax.set_ylim(0.2, 1.2)
ax.grid()
ax.set_xlabel("Model Size in Bytes")
ax.set_ylabel("F1-Score")
ax.legend(loc="lower left")

ax.xaxis.set_major_formatter(mkformatter)

plt.savefig(f"../figures/pareto-data-aware.png", format="png")

# Plot fixed input experiment:
plt.close("all")
_, ax = plt.subplots()

ax.plot(run21["Model Size"], run21["F1"], "xk", label="Run 1")
ax.plot(run22["Model Size"], run22["F1"], "+k", label="Run 2")
ax.plot(run23["Model Size"], run23["F1"], "ok", label="Run 3")
ax.plot(run24["Model Size"], run24["F1"], "vk", label="Run 4")
ax.plot(run25["Model Size"], run25["F1"], "sk", label="Run 5")

ax.set_xlim(20000000, 7000000)

ax.set_ylim(0.2, 1.2)
ax.grid()
ax.set_xlabel("Model Size in Bytes")
ax.set_ylabel("F1-Score")
ax.legend(loc="lower left")

ax.xaxis.set_major_formatter(mkformatter)

plt.savefig(f"../figures/pareto-fixed-data.png", format="png")
