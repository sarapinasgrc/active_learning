from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from src.utils.initialization import initialization_minimum, initialization_mean
from src.main import active_leaning_direct
from src.utils.preprocessing import preprocess_results
import matplotlib.pyplot as plt

file_path = "path"
dataset = pd.read_excel(file_path, sheet_name="Sheet1")

X = dataset.drop(columns=["label"]).values
Y = dataset["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

# ---- INITIALIZATION ----
distance_differences_min, elbow_iteration_min, elbow_point_min, selected_indeces = initialization_minimum(X_train)
#distance_differences_min, elbow_iteration_min, elbow_point_min, selected_indeces = initialization_mean(X_train)


# ---- PRINT ELBOW RESULTS ----
print(selected_indeces)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(distance_differences_min) + 1), distance_differences_min, marker='o', linestyle='-', color='blue', label="Distance Differences")
if elbow_iteration_min is not None:
    plt.scatter(elbow_iteration_min, elbow_point_min, color='red', s=100, label=f"Elbow Point at iteration {elbow_iteration_min} with value {elbow_point_min:.2f}", zorder=3)
    print(f"Elbow Point for K-Center Greedy Sampling at Iteration {elbow_iteration_min}, with value {elbow_point_min}")

plt.title("K-Center Greedy Sampling Initialization, No Labeled Data")
plt.xlabel("Iteration")
plt.ylabel("Distance Difference")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ---- ACTIVE LEARNING LOOP WITH SELECTED INIDICES ----

X_initial = X_train[selected_indeces]
print(X_initial)
print(X_initial.shape)
y_initial = y_train[selected_indeces]
mask = np.ones(len(X_train), dtype=bool)
mask[selected_indeces] = False
X_unlabeled = X_train[mask]
y_unlabeled = y_train[mask]

methods = ["coreset"]
colors = {"coreset": "orange", "entropy": "orange", "euclidean": "red", "support": "violet", "marginsampling": "mediummorchid", "leastconfident": "darkorange", "uncertainty_distance": "green"}

plt.figure(figsize=(14, 8))
x_range = range(1, 141)

results_single_init = {
    "accuracy": {},
    "auc": {},
    "methods": methods,
    "x_range": list(x_range),
    "size_initial": len(X_initial),
    "initialization": 'initialization used',
    "n_clusters": None,
}

for method in methods:
    print(f"Running method: {method}")
    results = active_leaning_direct(
        X_initial, y_initial, X_unlabeled, y_unlabeled, X_test, y_test, al=method, model="svm"
    )
    
    accuracy, auc = preprocess_results(results)
    auc = np.pad(auc, (140 - len(auc), 0), 'constant')
    accuracy = np.pad(accuracy, (140 - len(accuracy), 0), 'constant')
    results_single_init["accuracy"][method] = accuracy
    results_single_init["auc"][method] = auc

    if method == "coreset":
        plt.plot(
            x_range[:len(accuracy)], accuracy, label=f"Accuracy Euclidean + Entropy Strategy", color=colors[method], marker="o"
        )

plt.xlabel("Amount of Labeled Data")
plt.ylabel("Accuracy")
plt.title(f"Plot as Example")
plt.legend()
plt.grid(True)
plt.show() 


