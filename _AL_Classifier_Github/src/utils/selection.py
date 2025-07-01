from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

# Same logic as the initialization functions, but integrated in the active learning loop. Needs beforehand the amount of samples to select.
# I implemented the elbow detection method only on the initialization.py file.

def minimum_distance(X_unlabeled, y_unlabeled, X_initial, y_initial, n_samples=1):
  '''
  Selects the maximum minimum distance point from the unlabeled set at each iteration.
  '''
  if len(X_initial) == 0:
    mean_point = np.mean(X_unlabeled, axis=0)
    initial_idx = np.argmin(np.linalg.norm(X_unlabeled - mean_point, axis=1))

    selected_indices = [initial_idx]
    X_initial = [X_unlabeled[initial_idx]]
    y_initial = [y_unlabeled[initial_idx]]

    mask = np.ones(len(X_unlabeled), dtype=bool)
    mask[initial_idx] = False
    X_unlabeled = X_unlabeled[mask]
    y_unlabeled = y_unlabeled[mask]


  for _ in range(n_samples - 1):

    min_distances = []
    for i, x in enumerate(X_unlabeled):
        distances = euclidean_distances([x], X_initial)
        min_distance = np.min(distances)
        min_distances.append((i, min_distance))

    best_idx, best_min_distance = max(min_distances, key=lambda x: x[1])

    selected_indices.append(best_idx)
    X_initial = np.vstack([X_initial, X_unlabeled[best_idx]])
    y_initial = np.append(y_initial, y_unlabeled[best_idx])

    mask = np.ones(len(X_unlabeled), dtype=bool)
    mask[best_idx] = False
    X_unlabeled = X_unlabeled[mask]
    y_unlabeled = y_unlabeled[mask]

  print(f"Selected indexes: {selected_indices}")
  return X_unlabeled, y_unlabeled, X_initial, y_initial


def mean_distance(X_unlabeled, y_unlabeled, X_initial=None, y_initial=None, n_samples=1):
  '''
  Selects the maximum mean distance point from the unlabeled set at each iteration.
  '''
  if X_initial is None or len(X_initial) == 0:
        X_initial = []
        y_initial = []

  if len(X_initial) == 0:
        mean_point = np.mean(X_unlabeled, axis=0)
        initial_idx = np.argmin(np.linalg.norm(X_unlabeled - mean_point, axis=1))
        X_initial = [X_unlabeled[initial_idx]]
        y_initial = [y_unlabeled[initial_idx]]
        mask = np.ones(len(X_unlabeled), dtype=bool)
        mask[initial_idx] = False
        X_unlabeled = X_unlabeled[mask]
        y_unlabeled = y_unlabeled[mask]

  for _ in range(n_samples - 1):

    avg_distances = []

    for i, x in enumerate(X_unlabeled):
        distances = euclidean_distances([x], X_initial)
        mean_distance = np.mean(distances)
        avg_distances.append((i, mean_distance))

    best_idx, best_avg_distance = max(avg_distances, key=lambda x: x[1])

    X_initial = np.vstack([X_initial, X_unlabeled[best_idx]])
    y_initial = np.append(y_initial, y_unlabeled[best_idx])
    mask = np.ones(len(X_unlabeled), dtype=bool)
    mask[best_idx] = False
    X_unlabeled = X_unlabeled[mask]
    y_unlabeled = y_unlabeled[mask]

  return X_unlabeled, y_unlabeled, X_initial, y_initial