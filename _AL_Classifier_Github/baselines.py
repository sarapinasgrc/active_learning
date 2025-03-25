import numpy as np
import matplotlib.pyplot as plt
from src import MultiClassTSVM
from sklearn.metrics import accuracy_score, roc_auc_score 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.gaussian_process import GaussianProcessClassifier, RandomForestClassifier, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


def baseline_gpc(data, target_column, n_repeats=10, test_size=0.3):
    """
    Iteratively chooses a random index to train a GPClassifier model.

    Inputs:
    - data: pandas DataFrame with the data.
    - target_column: string with the name of the target column.
    - n_repeats: int with the number of repetitions.
    - test_size: float with the proportion of the test set.

    Outputs:
    - results: dictionary with the results of the experiment.
        - accuracy_per_iteration: list with the accuracy per iteration and per experiment.
        - auc_per_iteration: list with the AUC per iteration and per experiment.
        - accuracy_means: list with the mean accuracy per iteration.
        - auc_means: list with the mean AUC per iteration.
    """

    X = data.drop(columns=[target_column])
    Y = data[target_column]

    accuracy_per_iteration = []
    auc_per_iteration = []

    for repeat in range(n_repeats):

        X = np.array(X)
        Y = np.array(Y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=test_size, random_state=42, stratify=Y
        )
        
        y_test_bin = label_binarize(y_test, classes=np.unique(Y))

        random_indices = np.random.choice(len(X_train), size=2, replace=False)
        X_initial = X_train[random_indices]
        y_initial = y_train[random_indices]
        mask = np.ones(len(X_train), dtype=bool)
        mask[random_indices] = False
        X_unlabeled = X_train[mask]
        y_unlabeled = y_train[mask]

        accuracy_baseline = []
        auc_baseline = []

        while len(np.unique(y_initial)) < 2:

            accuracy_baseline.append(0)
            auc_baseline.append(0)

            new_index = np.random.choice(len(X_unlabeled), size=1, replace=False)[0]
            new_sample = X_unlabeled[new_index]
            new_label = y_unlabeled[new_index]

            X_initial = np.vstack((X_initial, new_sample))
            y_initial = np.append(y_initial, new_label)

            X_unlabeled = np.delete(X_unlabeled, new_index, axis=0)
            y_unlabeled = np.delete(y_unlabeled, new_index)

        y_test = np.array(y_test)

        kernel = DotProduct() + WhiteKernel()
        gpc = GaussianProcessClassifier(kernel=kernel, random_state=42)

        while len(X_unlabeled) > 0:

                  gpc.fit(X_initial, y_initial)
                  y_pred_test = gpc.predict(X_test)
                  y_pred_proba_test = gpc.predict_proba(X_test)[:, 1]

                  n_classes = len(np.unique(y_test))

                  if n_classes > 2:
                        auc = roc_auc_score(y_test_bin, y_pred_proba_test, multi_class='ovr')
                  else:
                        auc = roc_auc_score(y_test, y_pred_proba_test[:, 1])

                  accuracy = accuracy_score(y_test, y_pred_test)

                  print(f"Accuracy: {accuracy}, AUC: {auc}")

                  accuracy_baseline.append(accuracy)
                  auc_baseline.append(auc)

                  random_idx = np.random.choice(X_unlabeled.shape[0])

                  X_initial = np.vstack([X_initial, X_unlabeled[random_idx]])
                  y_initial = np.append(y_initial, y_unlabeled[random_idx])
                  X_unlabeled = np.delete(X_unlabeled, random_idx, axis=0)
                  y_unlabeled = np.delete(y_unlabeled, random_idx)


        accuracy_per_iteration.append(accuracy_baseline)
        auc_per_iteration.append(auc_baseline)

    plt.figure(figsize=(12, 8))
    for i in range(n_repeats):
        plt.plot(range(1, len(accuracy_per_iteration[i]) + 1), accuracy_per_iteration[i], label=f"Repetition {i + 1} - Accuracy", marker="o")

    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Iteration (GPClassifier)")
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(12, 8))
    for i in range(n_repeats):
        plt.plot(range(1, len(auc_per_iteration[i]) + 1), auc_per_iteration[i], label=f"Repetition {i + 1} - AUC", marker="o")

    plt.xlabel("Iterations")
    plt.ylabel("AUC")
    plt.title("AUC per Iteration (GPClassifier)")
    plt.legend()
    plt.grid(True)
    plt.show()

    results = {
        "accuracy_per_iteration": accuracy_per_iteration,
        "auc_per_iteration": auc_per_iteration,
        "accuracy_means": [np.mean(acc) for acc in accuracy_per_iteration],
        "auc_means": [np.mean(auc) for auc in auc_per_iteration]
    }

    return results

def baseline_logreg(data, target_column, n_repeats=10, test_size=0.3):
    """
    Iteratively chooses a random index to train a LogReg model.

    Inputs:
    - data: pandas DataFrame with the data.
    - target_column: string with the name of the target column.
    - n_repeats: int with the number of repetitions.
    - test_size: float with the proportion of the test set.

    Outputs:
    - results: dictionary with the results of the experiment.
        - accuracy_per_iteration: list with the accuracy per iteration and per experiment.
        - auc_per_iteration: list with the AUC per iteration and per experiment.
        - accuracy_means: list with the mean accuracy per iteration.
        - auc_means: list with the mean AUC per iteration.
    """

    X = data.drop(columns=[target_column])
    Y = data[target_column]

    accuracy_per_iteration = []
    auc_per_iteration = []

    for repeat in range(n_repeats):

        X = np.array(X)
        Y = np.array(Y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=test_size, random_state=42, stratify=Y
        )
        y_test_bin = label_binarize(y_test, classes=np.unique(Y))

        random_indices = np.random.choice(len(X_train), size=2, replace=False)
        X_initial = X_train[random_indices]
        y_initial = y_train[random_indices]
        mask = np.ones(len(X_train), dtype=bool)
        mask[random_indices] = False
        X_unlabeled = X_train[mask]
        y_unlabeled = y_train[mask]

        accuracy_baseline = []
        auc_baseline = []

        while len(np.unique(y_initial)) < 2:

            accuracy_baseline.append(0)
            auc_baseline.append(0)

            new_index = np.random.choice(len(X_unlabeled), size=1, replace=False)[0]
            new_sample = X_unlabeled[new_index]
            new_label = y_unlabeled[new_index]

            X_initial = np.vstack((X_initial, new_sample))
            y_initial = np.append(y_initial, new_label)

            X_unlabeled = np.delete(X_unlabeled, new_index, axis=0)
            y_unlabeled = np.delete(y_unlabeled, new_index)

        y_test = np.array(y_test)

        logreg = LogisticRegression(random_state=42)


        while len(X_unlabeled) > 0:

            logreg.fit(X_initial, y_initial)
            y_pred_test = logreg.predict(X_test)
            y_pred_proba_test = logreg.predict_proba(X_test)[:, 1]


            n_classes = len(np.unique(y_test))

            if n_classes > 2:
                auc = roc_auc_score(y_test_bin, y_pred_proba_test, multi_class='ovr')
            else:
                auc = roc_auc_score(y_test, y_pred_proba_test[:, 1])

            accuracy = accuracy_score(y_test, y_pred_test)

            accuracy_baseline.append(accuracy)
            auc_baseline.append(auc)

            random_idx = np.random.choice(X_unlabeled.shape[0])

            X_initial = np.vstack([X_initial, X_unlabeled[random_idx]])
            y_initial = np.append(y_initial, y_unlabeled[random_idx])
            X_unlabeled = np.delete(X_unlabeled, random_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, random_idx)


        accuracy_per_iteration.append(accuracy_baseline)
        auc_per_iteration.append(auc_baseline)

    plt.figure(figsize=(12, 8))
    for i in range(n_repeats):
        plt.plot(range(1, len(accuracy_per_iteration[i]) + 1), accuracy_per_iteration[i], label=f"Repetition {i + 1} - Accuracy", marker="o")

    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Iteration (LogReg)")
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(12, 8))
    for i in range(n_repeats):
        plt.plot(range(1, len(auc_per_iteration[i]) + 1), auc_per_iteration[i], label=f"Repetition {i + 1} - AUC", marker="o")

    plt.xlabel("Iterations")
    plt.ylabel("AUC")
    plt.title("AUC per Iteration (LogReg)")
    plt.legend()
    plt.grid(True)
    plt.show()

    results = {
        "accuracy_per_iteration": accuracy_per_iteration,
        "auc_per_iteration": auc_per_iteration,
        "accuracy_means": [np.mean(acc) for acc in accuracy_per_iteration],
        "auc_means": [np.mean(auc) for auc in auc_per_iteration]
    }

    return results

def baseline_rf(data, target_column, n_repeats=10, test_size=0.3):
    """
    Iteratively chooses a random index to train a Random Forest model.

    Inputs:
    - data: pandas DataFrame with the data.
    - target_column: string with the name of the target column.
    - n_repeats: int with the number of repetitions.
    - test_size: float with the proportion of the test set.

    Outputs:
    - results: dictionary with the results of the experiment.
        - accuracy_per_iteration: list with the accuracy per iteration and per experiment.
        - auc_per_iteration: list with the AUC per iteration and per experiment.
        - accuracy_means: list with the mean accuracy per iteration.
        - auc_means: list with the mean AUC per iteration.
    """

    X = data.drop(columns=[target_column])
    Y = data[target_column]

    accuracy_per_iteration = []
    auc_per_iteration = []

    for repeat in range(n_repeats):

        X = np.array(X)
        Y = np.array(Y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=test_size, random_state=42, stratify=Y
        )
        y_test_bin = label_binarize(y_test, classes=np.unique(Y))

        random_indices = np.random.choice(len(X_train), size=2, replace=False)
        X_initial = X_train[random_indices]
        y_initial = y_train[random_indices]
        mask = np.ones(len(X_train), dtype=bool)
        mask[random_indices] = False
        X_unlabeled = X_train[mask]
        y_unlabeled = y_train[mask]

        accuracy_baseline = []
        auc_baseline = []

        while len(np.unique(y_initial)) < 2:

            accuracy_baseline.append(0)
            auc_baseline.append(0)

            new_index = np.random.choice(len(X_unlabeled), size=1, replace=False)[0]
            new_sample = X_unlabeled[new_index]
            new_label = y_unlabeled[new_index]

            X_initial = np.vstack((X_initial, new_sample))
            y_initial = np.append(y_initial, new_label)

            X_unlabeled = np.delete(X_unlabeled, new_index, axis=0)
            y_unlabeled = np.delete(y_unlabeled, new_index)

        y_test = np.array(y_test)

        rf = RandomForestClassifier(random_state=42)


        while len(X_unlabeled) > 0:

            rf.fit(X_initial, y_initial)
            y_pred_test = rf.predict(X_test)
            y_pred_proba_test = rf.predict_proba(X_test)[:, 1]

            n_classes = len(np.unique(y_test))

            if n_classes > 2:
                auc = roc_auc_score(y_test_bin, y_pred_proba_test, multi_class='ovr')
            else:
                auc = roc_auc_score(y_test, y_pred_proba_test[:, 1])

            accuracy = accuracy_score(y_test, y_pred_test)

            accuracy_baseline.append(accuracy)
            auc_baseline.append(auc)

            random_idx = np.random.choice(X_unlabeled.shape[0])

            X_initial = np.vstack([X_initial, X_unlabeled[random_idx]])
            y_initial = np.append(y_initial, y_unlabeled[random_idx])
            X_unlabeled = np.delete(X_unlabeled, random_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, random_idx)


        accuracy_per_iteration.append(accuracy_baseline)
        auc_per_iteration.append(auc_baseline)

    plt.figure(figsize=(12, 8))
    for i in range(n_repeats):
        plt.plot(range(1, len(accuracy_per_iteration[i]) + 1), accuracy_per_iteration[i], label=f"Repetition {i + 1} - Accuracy", marker="o")

    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Iteration (Random Forest)")
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(12, 8))
    for i in range(n_repeats):
        plt.plot(range(1, len(auc_per_iteration[i]) + 1), auc_per_iteration[i], label=f"Repetition {i + 1} - AUC", marker="o")

    plt.xlabel("Iterations")
    plt.ylabel("AUC")
    plt.title("AUC per Iteration (Random Forest)")
    plt.legend()
    plt.grid(True)
    plt.show()

    results = {
        "accuracy_per_iteration": accuracy_per_iteration,
        "auc_per_iteration": auc_per_iteration,
        "accuracy_means": [np.mean(acc) for acc in accuracy_per_iteration],
        "auc_means": [np.mean(auc) for auc in auc_per_iteration]
    }

    return results


def baseline_svm(data, target_column, n_repeats=10, test_size=0.3):
    """
    Iteratively chooses a random index to train a SVM model.

    Inputs:
    - data: pandas DataFrame with the data.
    - target_column: string with the name of the target column.
    - n_repeats: int with the number of repetitions.
    - test_size: float with the proportion of the test set.

    Outputs:
    - results: dictionary with the results of the experiment.
        - accuracy_per_iteration: list with the accuracy per iteration and per experiment.
        - auc_per_iteration: list with the AUC per iteration and per experiment.
        - accuracy_means: list with the mean accuracy per iteration.
        - auc_means: list with the mean AUC per iteration.
    """

    X = data.drop(columns=[target_column])
    Y = data[target_column]

    accuracy_per_iteration = []
    auc_per_iteration = []

    for repeat in range(n_repeats):

        X = np.array(X)
        Y = np.array(Y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=test_size, random_state=42, stratify=Y
        )

        y_test_bin = label_binarize(y_test, classes=np.unique(Y))

        random_indices = np.random.choice(len(X_train), size=2, replace=False)
        X_initial = X_train[random_indices]
        y_initial = y_train[random_indices]
        mask = np.ones(len(X_train), dtype=bool)
        mask[random_indices] = False
        X_unlabeled = X_train[mask]
        y_unlabeled = y_train[mask]

        accuracy_baseline = []
        auc_baseline = []

        while len(np.unique(y_initial)) < 2:

            accuracy_baseline.append(0)
            auc_baseline.append(0)

            new_index = np.random.choice(len(X_unlabeled), size=1, replace=False)[0]
            new_sample = X_unlabeled[new_index]
            new_label = y_unlabeled[new_index]

            X_initial = np.vstack((X_initial, new_sample))
            y_initial = np.append(y_initial, new_label)

            X_unlabeled = np.delete(X_unlabeled, new_index, axis=0)
            y_unlabeled = np.delete(y_unlabeled, new_index)

        y_test = np.array(y_test)

        svm = SVC(kernel='linear', random_state=42, probability=True)

        while len(X_unlabeled) > 0:

            svm.fit(X_initial, y_initial)
            y_pred_test = svm.predict(X_test)
            y_pred_proba_test = svm.predict_proba(X_test)


            n_classes = len(np.unique(y_test))

            if n_classes > 2:
                auc = roc_auc_score(y_test_bin, y_pred_proba_test, multi_class='ovr')
            else:
                auc = roc_auc_score(y_test, y_pred_proba_test[:, 1])

            accuracy = accuracy_score(y_test, y_pred_test)

            accuracy_baseline.append(accuracy)
            auc_baseline.append(auc)

            random_idx = np.random.choice(X_unlabeled.shape[0])

            X_initial = np.vstack([X_initial, X_unlabeled[random_idx]])
            y_initial = np.append(y_initial, y_unlabeled[random_idx])
            X_unlabeled = np.delete(X_unlabeled, random_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, random_idx)



        accuracy_per_iteration.append(accuracy_baseline)
        auc_per_iteration.append(auc_baseline)

    plt.figure(figsize=(12, 8))
    for i in range(n_repeats):
        plt.plot(range(1, len(accuracy_per_iteration[i]) + 1), accuracy_per_iteration[i], label=f"Repetition {i + 1} - Accuracy", marker="o")

    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Iteration (SVM)")
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(12, 8))
    for i in range(n_repeats):
        plt.plot(range(1, len(auc_per_iteration[i]) + 1), auc_per_iteration[i], label=f"Repetition {i + 1} - AUC", marker="o")

    plt.xlabel("Iterations")
    plt.ylabel("AUC")
    plt.title("AUC per Iteration (SVM)")
    plt.legend()
    plt.grid(True)
    plt.show()

    results = {
        "accuracy_per_iteration": accuracy_per_iteration,
        "auc_per_iteration": auc_per_iteration,
        "accuracy_means": [np.mean(acc) for acc in accuracy_per_iteration],
        "auc_means": [np.mean(auc) for auc in auc_per_iteration]
    }

    return results


def baseline_tsvm(data, target_column, n_repeats=10, test_size=0.3):
    """
    Iteratively chooses a random index to train a TSVM model.

    Inputs:
    - data: pandas DataFrame with the data.
    - target_column: string with the name of the target column.
    - n_repeats: int with the number of repetitions.
    - test_size: float with the proportion of the test set.

    Outputs:
    - results: dictionary with the results of the experiment.
        - accuracy_per_iteration: list with the accuracy per iteration and per experiment.
        - auc_per_iteration: list with the AUC per iteration and per experiment.
        - accuracy_means: list with the mean accuracy per iteration.
        - auc_means: list with the mean AUC per iteration.
    """

    X = data.drop(columns=[target_column])
    Y = data[target_column]

    accuracy_per_iteration = []
    auc_per_iteration = []

    for repeat in range(n_repeats):

        print(f"Experiment {repeat + 1}/{n_repeats}")

        X = np.array(X)
        Y = np.array(Y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=test_size, random_state=42, stratify=Y
        )
        print("Train size:", len(X_train))
        y_test_bin = label_binarize(y_test, classes=np.unique(Y))

        random_indices = np.random.choice(len(X_train), size=2, replace=False)
        X_initial = X_train[random_indices]
        y_initial = y_train[random_indices]
        mask = np.ones(len(X_train), dtype=bool)
        mask[random_indices] = False
        X_unlabeled = X_train[mask]
        y_unlabeled = y_train[mask]

        accuracy_baseline = []
        auc_baseline = []

        while len(np.unique(y_initial)) < 2:

            accuracy_baseline.append(0)
            auc_baseline.append(0)

            new_index = np.random.choice(len(X_unlabeled), size=1, replace=False)[0]
            new_sample = X_unlabeled[new_index]
            new_label = y_unlabeled[new_index]

            X_initial = np.vstack((X_initial, new_sample))
            y_initial = np.append(y_initial, new_label)

            X_unlabeled = np.delete(X_unlabeled, new_index, axis=0)
            y_unlabeled = np.delete(y_unlabeled, new_index)

        y_test = np.array(y_test)

        tsvm = MultiClassTSVM(kernel='linear')

        while len(X_unlabeled) > 0:

            tsvm.fit(X_initial, y_initial, X_unlabeled)
            y_pred_test = tsvm.predict(X_test)
            y_pred_proba_test = tsvm.predict_proba(X_test)
  
            n_classes = len(np.unique(y_test))

            if n_classes > 2:
                auc = roc_auc_score(y_test_bin, y_pred_proba_test, multi_class='ovr')
            else:
                auc = roc_auc_score(y_test, y_pred_proba_test[:, 1])

            accuracy = accuracy_score(y_test, y_pred_test)

            accuracy_baseline.append(accuracy)
            auc_baseline.append(auc)

            random_idx = np.random.choice(X_unlabeled.shape[0])

            X_initial = np.vstack([X_initial, X_unlabeled[random_idx]])
            y_initial = np.append(y_initial, y_unlabeled[random_idx])
            X_unlabeled = np.delete(X_unlabeled, random_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, random_idx)


        accuracy_per_iteration.append(accuracy_baseline)
        auc_per_iteration.append(auc_baseline)

    plt.figure(figsize=(12, 8))
    for i in range(n_repeats):
        plt.plot(range(1, len(accuracy_per_iteration[i]) + 1), accuracy_per_iteration[i], label=f"Repetition {i + 1} - Accuracy", marker="o")

    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Iteration (TSVM)")
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(12, 8))
    for i in range(n_repeats):
        plt.plot(range(1, len(auc_per_iteration[i]) + 1), auc_per_iteration[i], label=f"Repetition {i + 1} - AUC", marker="o")

    plt.xlabel("Iterations")
    plt.ylabel("AUC")
    plt.title("AUC per Iteration (TSVM)")
    plt.legend()
    plt.grid(True)
    plt.show()

    results = {
        "accuracy_per_iteration": accuracy_per_iteration,
        "auc_per_iteration": auc_per_iteration,
        "accuracy_means": [np.mean(acc) for acc in accuracy_per_iteration],
        "auc_means": [np.mean(auc) for auc in auc_per_iteration]
    }

    return results
