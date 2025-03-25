import numpy as np

def preprocess_results(results, target_length=140):
    """
    Preprocesses the results of the active learning experiments.

    Parameters:
    - results: dictionary with the results of the active learning experiment.

    Outputs:
    - accuracy: array with the accuracy per iteration.
    - auc: array with the AUC per iteration.
    """

    if "accuracy_list" in results and "auc_list" in results:
        accuracy = results["accuracy_list"]
        auc = results["auc_list"]

        while len(accuracy) < target_length:
            accuracy.insert(0, 0)
        while len(auc) < target_length:
            auc.insert(0, 0)

        accuracy = np.array(accuracy)
        auc = np.array(auc)
        accuracy[accuracy == 0] = np.nan
        auc[auc == 0] = np.nan

    else:
        accuracy_per_iteration = results['accuracy_per_iteration']
        auc_per_iteration = results['auc_per_iteration']
        min_len = min(len(iteration) for iteration in accuracy_per_iteration)
        min_len_auc = min(len(iteration) for iteration in auc_per_iteration)
        truncated_accuracy_per_iteration = [iteration[:min_len] for iteration in accuracy_per_iteration]
        truncated_auc_per_iteration = [iteration[:min_len_auc] for iteration in auc_per_iteration]
        mean_accuracy = np.mean(truncated_accuracy_per_iteration, axis=0)
        mean_auc_base = np.mean(truncated_auc_per_iteration, axis=0)

        accuracy = mean_accuracy.tolist()
        auc = mean_auc_base.tolist()

        while len(accuracy) < 140:
            accuracy.insert(0, 0)

        while len(auc) < 140:
            auc.insert(0, 0)

        accuracy = np.array(accuracy)
        accuracy[accuracy == 0] = np.nan
        auc = np.array(auc)
        auc[auc == 0] = np.nan

    return accuracy, auc


def preprocess_mean_and_std(accuracy_iterations, target_length=140):
    """
    Preprocesses the mean and standard deviation of the accuracy over multiple experiments.
    
    Parameters:
    - accuracy_iterations: list with the accuracy per iteration and per experiment.
    
    Outputs:
    - mean_accuracy: array with the mean accuracy per iteration.
    """

    min_len = min(len(it) for it in accuracy_iterations)
    truncated_iterations = [it[:min_len] for it in accuracy_iterations]

    mean_accuracy = np.mean(truncated_iterations, axis=0).tolist()
    std_accuracy = np.std(truncated_iterations, axis=0).tolist()
    while len(mean_accuracy) < target_length:
        mean_accuracy.insert(0, 0)
    while len(std_accuracy) < target_length:
        std_accuracy.insert(0, 0)

    mean_accuracy = np.array(mean_accuracy)
    std_accuracy = np.array(std_accuracy)
    mean_accuracy[mean_accuracy == 0] = np.nan
    std_accuracy[std_accuracy == 0] = np.nan

    return mean_accuracy, std_accuracy