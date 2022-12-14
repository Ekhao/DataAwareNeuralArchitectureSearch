import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np

run1 = pd.read_csv(
    "experiment/results/evolution_exp1_run1.csv", index_col="Model Number")
run2 = pd.read_csv(
    "experiment/results/evolution_exp1_run2.csv", index_col="Model Number")
run3 = pd.read_csv(
    "experiment/results/evolution_exp1_run3.csv", index_col="Model Number")
run4 = pd.read_csv(
    "experiment/results/evolution_exp1_run4.csv", index_col="Model Number")
run5 = pd.read_csv(
    "experiment/results/evolution_exp1_run5.csv", index_col="Model Number")

# Calculate their fitness function


def add_fitness_function(row):
    model_size_score = math.exp(-row["Model Size"] /
                                100000)
    return row["Accuracy"] + \
        row["Precision"] + row["Recall"] + model_size_score


run1["Fitness"] = run1.apply(lambda row: add_fitness_function(row), axis=1)
run2["Fitness"] = run2.apply(lambda row: add_fitness_function(row), axis=1)
run3["Fitness"] = run3.apply(lambda row: add_fitness_function(row), axis=1)
run4["Fitness"] = run4.apply(lambda row: add_fitness_function(row), axis=1)
run5["Fitness"] = run5.apply(lambda row: add_fitness_function(row), axis=1)

# Generate x axises
x_values_run1 = range(1, len(run1["Fitness"]) + 1, 20)
x_values_run2 = range(1, len(run2["Fitness"]) + 1, 20)
x_values_run3 = range(1, len(run3["Fitness"]) + 1, 20)
x_values_run4 = range(1, len(run4["Fitness"]) + 1, 20)
x_values_run5 = range(1, len(run5["Fitness"]) + 1, 20)

# Calculate average values of a number of rows to reduce clutter
run1 = run1.groupby(np.arange(len(run1))//20).mean(numeric_only=True)
run2 = run2.groupby(np.arange(len(run2))//20).mean(numeric_only=True)
run3 = run3.groupby(np.arange(len(run3))//20).mean(numeric_only=True)
run4 = run4.groupby(np.arange(len(run4))//20).mean(numeric_only=True)
run5 = run5.groupby(np.arange(len(run5))//20).mean(numeric_only=True)

plt.close("all")
_, ax = plt.subplots()
ax.plot(x_values_run1, run1["Fitness"], label="Run 1")
ax.plot(x_values_run2, run2["Fitness"], label="Run 2")
ax.plot(x_values_run3, run3["Fitness"], label="Run 3")
ax.plot(x_values_run4, run4["Fitness"], label="Run 4")
ax.plot(x_values_run5, run5["Fitness"], label="Run 5")
ax.set_xlim(0, 300)
ax.grid()
ax.set_ylim(0, 4)
ax.set_xlabel("# of models evaluated")
ax.set_ylabel("Performance Estimation Score")
ax.legend(loc="lower right")
plt.savefig(f"figures/evolution-exp1.png", format="png")
