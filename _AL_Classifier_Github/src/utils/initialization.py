import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from kneed import KneeLocator

def initialization_minimum_manual(X_unlabeled, y_unlabeled, X_initial=None, y_initial=None, elbow=True, patience=2, tol=80):

    """
    Iteratively selects points based on the maximum minimum distance to the points already labeled. The stopping criteria 
    is manually set by the user.

    Params: 
    - X_unlabeled: numpy array with the unlabeled data.
    - y_unlabeled: numpy array with the labels of the unlabeled data.
    - X_initial: numpy array with the labeled data.
    - y_initial: numpy array with the labels of the labeled data.
    - elbow: boolean indicating whether to find the elbow point.
    - patience: int with the number of iterations to wait before stopping the algorithm.
    - tol: float with the tolerance to consider a point as an elbow point.

    Returns:
    - distance_differences: list with the maximum minimum distances per iteration.
    - elbow_iteration: int with the iteration where the elbow point was found.
    - elbow_point: float with the value of the elbow point.

    """
    
    distance_differences = []

    if X_initial is None or len(X_initial) == 0:
        X_initial = []
        y_initial = []

    iteration = 0

    if elbow:
      no_improve_counter = 0
      elbow_point = None
      elbow_iteration = None
      point_found = False

    while len(X_unlabeled) > 0:

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

      else:
        selected_indices = []

      min_distances = []
      for i, x in enumerate(X_unlabeled):
          distances = euclidean_distances([x], X_initial)
          min_distance = np.min(distances)
          min_distances.append((i, min_distance))


      best_idx, best_min_distance = max(min_distances, key=lambda x: x[1])

      if len(distance_differences) > 0:

          if len(distance_differences) > 0:

            if best_min_distance > distance_differences[-1] + tol or best_min_distance < distance_differences[-1] - tol:
                no_improve_counter = 0
            else:
                if no_improve_counter == 0:
                    elbow_point_candidate = best_min_distance
                    elbow_iteration_candidate = iteration

                no_improve_counter += 1

            if no_improve_counter >= patience and not point_found:
                elbow_point = elbow_point_candidate
                elbow_iteration = elbow_iteration_candidate
                point_found = True

      distance_differences.append(best_min_distance)
      iteration += 1
      print(f"For iteration {iteration}, best minimum distance: {best_min_distance}")

      selected_indices.append(best_idx)
      X_initial = np.vstack([X_initial, X_unlabeled[best_idx]])
      y_initial = np.append(y_initial, y_unlabeled[best_idx])
      mask = np.ones(len(X_unlabeled), dtype=bool)
      mask[best_idx] = False
      X_unlabeled = X_unlabeled[mask]
      y_unlabeled = y_unlabeled[mask]

      print(f"Len X_initial: {len(X_initial)}")
      print(f"Len X_unlabeled: {len(X_unlabeled)}")
      print("\n")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(distance_differences) + 1), distance_differences, marker='o', linestyle='-', color='b')
    if elbow_point is not None:
        plt.scatter(elbow_iteration + 1, elbow_point, color='red', s=100, label="Elbow Point", zorder=3)
        print(f"Elbow Point at iteration {elbow_iteration + 1}, value: {elbow_point}")
    else:
        print("No Elbow Point detected.")
    plt.title("Distance Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Maximum Minimmum distance")
    plt.grid(alpha=0.3)
    plt.show()

    return distance_differences, elbow_iteration, elbow_point

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

def initialization_minimum(X_unlabeled, X_initial=None):
    """
    Iteratively selects points based on the maximum minimum distance to the points already labeled. The stopping criteria
    is automatically set by the algorithm.
    
    Params:
    - X_unlabeled: numpy array with the unlabeled data.
    - X_initial: numpy array with the labeled data.
    
    Returns:
    - distance_differences: list with the maximum minimum distances per iteration.
    - elbow_iteration: int with the iteration where the elbow point was found.
    - elbow_point: float with the value of the elbow point.
    """

    distance_differences = []

    if X_initial is None or len(X_initial) == 0:
        X_initial = []

    iteration = 0

    initial_distances = []
    if len(X_initial) > 1:
        mean_point = np.mean(X_initial, axis=0)
        initial_idx = np.argmin(np.linalg.norm(X_initial - mean_point, axis=1))
        print(f"Initial index: {initial_idx}")
        X_aux = [X_initial[initial_idx]]
        X_rest = np.delete(X_initial, initial_idx, axis=0)
        while len(X_rest) > 0: 
            print(f"X_aux: {X_aux}")
            print(f"len(X_rest): {len(X_rest)}") 
            print(f"len(X_aux): {len(X_aux)}")    
            
            aux = []
            for i, x in enumerate(X_rest):
                distances = euclidean_distances([x], X_aux)
                min_distance = np.partition(distances, 1)[0,1] if np.min(distances) == 0 else np.min(distances)
                aux.append((i, min_distance))

            best_idx, best_min_distance = max(aux, key=lambda x: x[1])
            print(f"Best min distance: {best_min_distance}")

            initial_distances.append(best_min_distance)
    
            X_aux = np.vstack([X_aux, X_rest[best_idx]])
            mask = np.ones(len(X_rest), dtype=bool)
            mask[best_idx] = False
            X_rest = X_rest[mask]
    

    X_whole = np.vstack([X_initial, X_unlabeled])
    mean_point = np.mean(X_whole, axis=0)
    initial_idx = np.argmin(np.linalg.norm(X_whole - mean_point, axis=1))
    X_all = [X_whole[initial_idx]]

    mask = np.ones(len(X_whole), dtype=bool)
    mask[initial_idx] = False
    X_whole = X_whole[mask]
    distance_differences_whole = []
    while len(X_whole) > 0:

      min_distances_whole = []
      for i, x in enumerate(X_whole):
          distances = euclidean_distances([x], X_all)
          min_distance_whole = np.min(distances)
          min_distances_whole.append((i, min_distance_whole))

      best_idx, best_min_distance = max(min_distances_whole, key=lambda x: x[1])
      distance_differences_whole.append(best_min_distance)
      X_all = np.vstack([X_all, X_whole[best_idx]])
      mask = np.ones(len(X_whole), dtype=bool)
      mask[best_idx] = False
      X_whole = X_whole[mask]


    while len(X_unlabeled) > 0:

      if len(X_initial) == 0:
        mean_point = np.mean(X_unlabeled, axis=0)
        initial_idx = np.argmin(np.linalg.norm(X_unlabeled - mean_point, axis=1))

        selected_indices = [initial_idx]
        X_initial = [X_unlabeled[initial_idx]]

        mask = np.ones(len(X_unlabeled), dtype=bool)
        mask[initial_idx] = False
        X_unlabeled = X_unlabeled[mask]

      else:
        selected_indices = []

      min_distances = []
      for i, x in enumerate(X_unlabeled):
          distances = euclidean_distances([x], X_initial)
          min_distance = np.partition(distances, 1)[0,1] if np.min(distances) == 0 else np.min(distances)
          min_distances.append((i, min_distance))

      best_idx, best_min_distance = max(min_distances, key=lambda x: x[1])

      distance_differences.append(best_min_distance)
      iteration += 1

      selected_indices.append(best_idx)
      X_initial = np.vstack([X_initial, X_unlabeled[best_idx]])
      mask = np.ones(len(X_unlabeled), dtype=bool)
      mask[best_idx] = False
      X_unlabeled = X_unlabeled[mask]


    elbow_finder = KneeLocator(range(1, len(distance_differences) + 1), distance_differences, curve="convex", direction="decreasing", S=1)
    elbow_iteration = elbow_finder.knee
    elbow_point = distance_differences[elbow_iteration - 1] if elbow_iteration is not None else None

    plt.figure(figsize=(10, 6))

    if initial_distances:
        plt.plot(range(1, len(initial_distances) + 1), initial_distances, marker='o', linestyle='--', color='orange', label="Initial X Distance")

    plt.plot(range(1, len(distance_differences) + 1), distance_differences, marker='o', linestyle='-', color='b', label="Distance")
    plt.plot(range(1, len(distance_differences_whole) + 1), distance_differences_whole, marker='o', linestyle='-', color='green', label="All Distance")

    if elbow_iteration is not None:
        plt.scatter(elbow_iteration, elbow_point, color='red', s=100, label="Elbow Point", zorder=3)
        print(f"Elbow Point found at iteration {elbow_iteration}, value: {elbow_point}")
    else:
        print("No clear Elbow Point detected.")

    plt.title("Distance Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Maximum Minimum distance")
    plt.grid(alpha=0.3)
    plt.show()

    return distance_differences, elbow_iteration, elbow_point

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------


def initialization_mean_manual(X_unlabeled, y_unlabeled, X_initial=None, y_initial=None, elbow=True, patience=2, tol=80):

    """
    Iteratively selects points based on the maximum mean distance to the points already labeled. The stopping criteria 
    is manually set by the user.

    Params: 
    - X_unlabeled: numpy array with the unlabeled data.
    - y_unlabeled: numpy array with the labels of the unlabeled data.
    - X_initial: numpy array with the labeled data.
    - y_initial: numpy array with the labels of the labeled data.
    - elbow: boolean indicating whether to find the elbow point.
    - patience: int with the number of iterations to wait before stopping the algorithm.
    - tol: float with the tolerance to consider a point as an elbow point.

    Returns:
    - distance_differences: list with the maximum minimum distances per iteration.
    - elbow_iteration: int with the iteration where the elbow point was found.
    - elbow_point: float with the value of the elbow point.

    """
    
    distance_differences = []

    if X_initial is None or len(X_initial) == 0:
        X_initial = []
        y_initial = []

    iteration = 0

    if elbow:

      no_improve_counter = 0
      elbow_point = None
      elbow_iteration = None
      point_found = False

    while len(X_unlabeled) > 0:
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
        else:
            selected_indices = []

        avg_distances = []
        for i, x in enumerate(X_unlabeled):
            distances = euclidean_distances([x], X_initial)
            mean_distance = np.mean(distances)
            avg_distances.append((i, mean_distance))

        best_idx, best_avg_distance = max(avg_distances, key=lambda x: x[1])

        if len(distance_differences) > 0:

          if len(distance_differences) > 0:

            if best_avg_distance > distance_differences[-1] + tol or best_avg_distance < distance_differences[-1] - tol:
                no_improve_counter = 0
            else:
                if no_improve_counter == 0:
                    elbow_point_candidate = best_avg_distance
                    elbow_iteration_candidate = iteration

                no_improve_counter += 1

            if no_improve_counter >= patience and not point_found:
                elbow_point = elbow_point_candidate
                elbow_iteration = elbow_iteration_candidate
                point_found = True

        distance_differences.append(best_avg_distance)

        iteration += 1

        print(f"For iteration {iteration}, best average distance: {best_avg_distance}")

        selected_indices.append(best_idx)
        X_initial = np.vstack([X_initial, X_unlabeled[best_idx]])
        y_initial = np.append(y_initial, y_unlabeled[best_idx])
        mask = np.ones(len(X_unlabeled), dtype=bool)
        mask[best_idx] = False
        X_unlabeled = X_unlabeled[mask]
        y_unlabeled = y_unlabeled[mask]

        print(f"Len X_initial: {len(X_initial)}")
        print(f"Len X_unlabeled: {len(X_unlabeled)}")

        print("\n")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(distance_differences) + 1), distance_differences, marker='o', linestyle='-', color='b')
    if elbow_point is not None:
        plt.scatter(elbow_iteration + 1, elbow_point, color='red', s=100, label="Elbow Point", zorder=3)
        print(f"Elbow Point found at iteration {elbow_iteration + 1}, value: {elbow_point}")
    else:
        print("No clear Elbow Point detected.")
    plt.title("Distance Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Maximum mean distance")
    plt.grid(alpha=0.3)
    plt.show()

    return distance_differences, elbow_iteration, elbow_point

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

def initialization_mean(X_unlabeled, X_initial=None):

    """
    Iteratively selects points based on the maximum mean distance to the points already labeled. The stopping criteria
    is automatically set by the algorithm.
    
    Params:
    - X_unlabeled: numpy array with the unlabeled data.
    - X_initial: numpy array with the labeled data.
    
    Returns:
    - distance_differences: list with the maximum minimum distances per iteration.
    - elbow_iteration: int with the iteration where the elbow point was found.
    - elbow_point: float with the value of the elbow point.
    """

    distance_differences = []

    if X_initial is None or len(X_initial) == 0:
        X_initial = []

    iteration = 0

    while len(X_unlabeled) > 0:
        if len(X_initial) == 0:
            mean_point = np.mean(X_unlabeled, axis=0)
            initial_idx = np.argmin(np.linalg.norm(X_unlabeled - mean_point, axis=1))
            selected_indices = [initial_idx]
            X_initial = [X_unlabeled[initial_idx]]
            mask = np.ones(len(X_unlabeled), dtype=bool)
            mask[initial_idx] = False
            X_unlabeled = X_unlabeled[mask]
        else:
            selected_indices = []

        avg_distances = []
        for i, x in enumerate(X_unlabeled):
            distances = euclidean_distances([x], X_initial)
            mean_distance = np.mean(distances)
            avg_distances.append((i, mean_distance))

        best_idx, best_avg_distance = max(avg_distances, key=lambda x: x[1])
        distance_differences.append(best_avg_distance)

        iteration += 1

        print(f"For iteration {iteration}, best average distance: {best_avg_distance}")

        selected_indices.append(best_idx)
        X_initial = np.vstack([X_initial, X_unlabeled[best_idx]])
        mask = np.ones(len(X_unlabeled), dtype=bool)
        mask[best_idx] = False
        X_unlabeled = X_unlabeled[mask]

        print(f"Len X_initial: {len(X_initial)}")
        print(f"Len X_unlabeled: {len(X_unlabeled)}")

        print("\n")

    elbow_finder = KneeLocator(range(1, len(distance_differences) + 1), distance_differences, curve="convex", direction="decreasing")
    elbow_iteration = elbow_finder.knee
    elbow_point = distance_differences[elbow_iteration - 1] if elbow_iteration is not None else None

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(distance_differences) + 1), distance_differences, marker='o', linestyle='-', color='b', label="Distance")

    if elbow_iteration is not None:
        plt.scatter(elbow_iteration, elbow_point, color='red', s=100, label="Elbow Point", zorder=3)
        print(f"Elbow Point found at iteration {elbow_iteration}, value: {elbow_point}")
    else:
        print("No clear Elbow Point detected.")
    plt.title("Distance Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Maximum mean distance")
    plt.grid(alpha=0.3)
    plt.show()

    return distance_differences, elbow_iteration, elbow_point