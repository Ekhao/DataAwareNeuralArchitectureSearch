import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np

run11 = pd.read_csv(
    "experiment/results/evolution_exp1_run1.csv", index_col="Model Number")
run12 = pd.read_csv(
    "experiment/results/evolution_exp1_run2.csv", index_col="Model Number")
run13 = pd.read_csv(
    "experiment/results/evolution_exp1_run3.csv", index_col="Model Number")
run14 = pd.read_csv(
    "experiment/results/evolution_exp1_run4.csv", index_col="Model Number")
run15 = pd.read_csv(
    "experiment/results/evolution_exp1_run5.csv", index_col="Model Number")

run21 = pd.read_csv(
    "experiment/results/evolution_exp2_run1.csv", index_col="Model Number")
run22 = pd.read_csv(
    "experiment/results/evolution_exp2_run2.csv", index_col="Model Number")
run23 = pd.read_csv(
    "experiment/results/evolution_exp2_run3.csv", index_col="Model Number")
run24 = pd.read_csv(
    "experiment/results/evolution_exp2_run4.csv", index_col="Model Number")
run25 = pd.read_csv(
    "experiment/results/evolution_exp2_run5.csv", index_col="Model Number")

# Calculate their fitness function


def add_fitness_function(row):
    model_size_score = math.exp(-row["Model Size"] /
                                100000)
    return row["Accuracy"] + \
        row["Precision"] + row["Recall"] + model_size_score


run11["Fitness"] = run11.apply(lambda row: add_fitness_function(row), axis=1)
run12["Fitness"] = run12.apply(lambda row: add_fitness_function(row), axis=1)
run13["Fitness"] = run13.apply(lambda row: add_fitness_function(row), axis=1)
run14["Fitness"] = run14.apply(lambda row: add_fitness_function(row), axis=1)
run15["Fitness"] = run15.apply(lambda row: add_fitness_function(row), axis=1)

run21["Fitness"] = run21.apply(lambda row: add_fitness_function(row), axis=1)
run22["Fitness"] = run22.apply(lambda row: add_fitness_function(row), axis=1)
run23["Fitness"] = run23.apply(lambda row: add_fitness_function(row), axis=1)
run24["Fitness"] = run24.apply(lambda row: add_fitness_function(row), axis=1)
run25["Fitness"] = run25.apply(lambda row: add_fitness_function(row), axis=1)


# Generate x axises
x_values_run11 = range(1, len(run11["Fitness"]) + 1, 20)
x_values_run12 = range(1, len(run12["Fitness"]) + 1, 20)
x_values_run13 = range(1, len(run13["Fitness"]) + 1, 20)
x_values_run14 = range(1, len(run14["Fitness"]) + 1, 20)
x_values_run15 = range(1, len(run15["Fitness"]) + 1, 20)

x_values_run21 = range(1, len(run21["Fitness"]) + 1, 20)
x_values_run22 = range(1, len(run22["Fitness"]) + 1, 20)
x_values_run23 = range(1, len(run23["Fitness"]) + 1, 20)
x_values_run24 = range(1, len(run24["Fitness"]) + 1, 20)
x_values_run25 = range(1, len(run25["Fitness"]) + 1, 20)

# Calculate average values of a number of rows to reduce clutter
run11 = run11.groupby(np.arange(len(run11))//20).max(numeric_only=True)
run12 = run12.groupby(np.arange(len(run12))//20).max(numeric_only=True)
run13 = run13.groupby(np.arange(len(run13))//20).max(numeric_only=True)
run14 = run14.groupby(np.arange(len(run14))//20).max(numeric_only=True)
run15 = run15.groupby(np.arange(len(run15))//20).max(numeric_only=True)

run21 = run21.groupby(np.arange(len(run21))//20).max(numeric_only=True)
run22 = run22.groupby(np.arange(len(run22))//20).max(numeric_only=True)
run23 = run23.groupby(np.arange(len(run23))//20).max(numeric_only=True)
run24 = run24.groupby(np.arange(len(run24))//20).max(numeric_only=True)
run25 = run25.groupby(np.arange(len(run25))//20).max(numeric_only=True)

plt.close("all")
_, ax = plt.subplots()

ax.plot(x_values_run11, run11["Fitness"], "-k", label="Run 1")
ax.plot(x_values_run12, run12["Fitness"], "--k", label="Run 2")
ax.plot(x_values_run13, run13["Fitness"], "-.k", label="Run 3")
ax.plot(x_values_run14, run14["Fitness"], ":k", label="Run 4")
ax.plot(x_values_run15, run15["Fitness"], ".-k", label="Run 5")

ax.plot(x_values_run21, run21["Fitness"], "-b", label="Run 1")
ax.plot(x_values_run22, run22["Fitness"], "--b", label="Run 2")
ax.plot(x_values_run23, run23["Fitness"], "-.b", label="Run 3")
ax.plot(x_values_run24, run24["Fitness"], ":b", label="Run 4")
ax.plot(x_values_run25, run25["Fitness"], ".-b", label="Run 5")

ax.set_xlim(0, 300)
ax.grid()
ax.set_ylim(0, 4)
ax.set_xlabel("# of Models Evaluated")
ax.set_ylabel("Performance Estimation Score")
ax.legend(loc="lower right")
plt.savefig(f"figures/evolution-combined.png", format="png")
