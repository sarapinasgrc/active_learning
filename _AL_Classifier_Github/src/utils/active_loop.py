import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances 
from models import define_model, TSVM
from scipy.stats import entropy


def al_loop(X_labeled, y_labeled, X_pool, y_pool, X_test, y_test, model, al='coreset', kernel='linear'):
    """
    Active learning loop
    
        Parameters:
            X_labeled (numpy array): Labeled data
            y_labeled (numpy array): Labels of labeled data
            X_pool (numpy array): Pool of unlabeled data
            y_pool (numpy array): Labels of pool of unlabeled data
            X_test (numpy array): Test data
            y_test (numpy array): Labels of test data
            model (str): Model to use
            al (str): Active learning strategy
            kernel (str): Kernel to use in model

        Returns:
            results (dict): Dictionary with accuracy and AUC
    
    """

    model = define_model(model, kernel)

    accuracy_list = []
    auc_list = []
    iteration = 0

    while len(X_pool) > 0:

        if isinstance(model, TSVM):
            model.fit(X_labeled, y_labeled, X_pool)
        else:
            model.fit(X_labeled, y_labeled)

        if isinstance(model, TSVM):
            y_test_pred = model.predict(X_test, Transductive=False)
            y_pred_proba_test = model.predict_proba(X_test, Transductive=False)[:, 1]
        else: 
            y_test_pred = model.predict(X_test)
            y_pred_proba_test = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_test_pred)
        auc = roc_auc_score(y_test, y_pred_proba_test)
        accuracy_list.append(accuracy)
        auc_list.append(auc)

        if isinstance(model, TSVM):
            probs = model.predict_proba(X_pool, Transductive=False)
        else: 
            probs = model.predict_proba(X_pool)

        if al == 'coreset':
          entropies = entropy(probs, axis=1)
          dists = euclidean_distances(X_pool, X_labeled)
          scaled_dists = dists * entropies.reshape(-1, 1) 
          min_scaled_dist = np.min(scaled_dists, axis=1)
          next_idx = np.argmax(min_scaled_dist)

        elif al == 'leastconfident':
          max_confidences = np.max(probs, axis=1)
          next_idx = np.argmin(max_confidences)

        elif al == 'marginsampling':
          sorted_probs = np.sort(probs, axis=1)
          margins = sorted_probs[:, -1] - sorted_probs[:, -2]
          next_idx = np.argmin(margins)

        elif al == 'entropy':
          entropies = entropy(probs, axis=1)
          next_idx = np.argmax(entropies)

        elif al == 'variance':
          y_proba_unlabeled = model.predict_proba(X_pool)
          variance_unlabeled = np.var(y_proba_unlabeled, axis=1)
          next_idx = np.argmax(variance_unlabeled)

        elif al == 'uncertainty_distance':
          uncertainties = 1 - np.max(probs, axis=1)
          dists = euclidean_distances(X_pool, X_labeled)
          scaled_dists = dists * uncertainties.reshape(-1, 1)
          min_scaled_dist = np.min(scaled_dists, axis=1)
          next_idx = np.argmax(min_scaled_dist)

        X_labeled = np.vstack([X_labeled, X_pool[next_idx]])
        y_labeled = np.append(y_labeled, y_pool[next_idx])
        X_pool = np.delete(X_pool, next_idx, axis=0)
        y_pool = np.delete(y_pool, next_idx)
        iteration += 1

    results = {
        "accuracy_list": accuracy_list,
        "auc_list": auc_list,
    }
    return results