import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances 
from models import define_model, TSVM, MultiClassTSVM
from scipy.stats import entropy
from sklearn.preprocessing import label_binarize
from utils.stopping_criteria import MU_stopping, OU_stopping, MEE_stopping
from sklearn.metrics import pairwise_distances

def al_loop(X_labeled, y_labeled, X_pool, y_pool, X_test, y_test, model, al='coreset', kernel='linear', threshold=0.15):
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
            results (dict): Dictionary with accuracy and AUC for learning curve creation
    
    """

    model = define_model(model, kernel)

    accuracy_list = []
    auc_list = []
    iteration = 0
    mu, ou, mee,  = False, False, False
    point_mu, point_ou, point_mee = -1, -1, -1

    while len(X_pool) > 0:

        if isinstance(model, (TSVM, MultiClassTSVM)):
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

        if len(np.unique(y_test)) > 2:
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            auc = roc_auc_score(y_test_bin, y_pred_proba_test, multi_class='ovr')
        else:
            auc = roc_auc_score(y_test, y_pred_proba_test[:, 1])

        accuracy_list.append(accuracy)
        auc_list.append(auc)

        if isinstance(model, TSVM):
            probs = model.predict_proba(X_pool, Transductive=False)
        else: 
            probs = model.predict_proba(X_pool)

        if iteration > 1:
          if not mu:
              mu = MU_stopping(probs, threshold=threshold)
              if mu: 
                  point_mu = iteration
          
          if not ou:
              ou = OU_stopping(probs, threshold=threshold)
              if ou:
                  point_ou = iteration
              
          if not mee:
              mee = MEE_stopping(probs, X_pool, threshold=threshold)
              if mee:
                  point_mee = iteration

        if al == 'coreset':
          """
          Active learning strategy that selects the next sample based on the Entropy + Euclidean Distance.
          """
          entropies = entropy(probs, axis=1)
          dists = euclidean_distances(X_pool, X_labeled)
          scaled_dists = dists * entropies.reshape(-1, 1) 
          min_scaled_dist = np.min(scaled_dists, axis=1)
          next_idx = np.argmax(min_scaled_dist)

        elif al == 'leastconfident':
          """
          Active learning strategy that selects the next sample based on the least confident prediction.
          """
          max_confidences = np.max(probs, axis=1)
          next_idx = np.argmin(max_confidences)

        elif al == 'marginsampling':
          """
          Active learning strategy that selects the next sample based on the margin between the two highest predicted probabilities.
          """
          sorted_probs = np.sort(probs, axis=1)
          margins = sorted_probs[:, -1] - sorted_probs[:, -2]
          next_idx = np.argmin(margins)

        elif al == 'entropy':
          """
          Active learning strategy that selects the next sample based on the entropy of the predicted probabilities.
          """
          entropies = entropy(probs, axis=1)
          next_idx = np.argmax(entropies)

        elif al == 'variance':
          """
          Active learning strategy that selects the next sample based on the variance of the predicted probabilities.
          """
          y_proba_unlabeled = model.predict_proba(X_pool)
          variance_unlabeled = np.var(y_proba_unlabeled, axis=1)
          next_idx = np.argmax(variance_unlabeled)

        elif al == 'uncertainty_distance':
          """
          Active learning strategy that selects the next sample based on Least Confident + Euclidean distance.
          """
          uncertainties = 1 - np.max(probs, axis=1)
          dists = euclidean_distances(X_pool, X_labeled)
          scaled_dists = dists * uncertainties.reshape(-1, 1)
          min_scaled_dist = np.min(scaled_dists, axis=1)
          next_idx = np.argmax(min_scaled_dist)

        elif al == 'support':
            """
            Active learning strategy that selects the next sample based on the unlabeled support vectors of the TSVM model.
            """
            if isinstance(model, (TSVM, MultiClassTSVM)):
                support_vectors = model.support_vectors_
                support_indices = [
                    i for i, x in enumerate(X_pool)
                    if any(np.allclose(x, sv, atol=1e-6) for sv in support_vectors)
                ]
                
                if support_indices:
                    next_idx = support_indices[0]

                else:
                    # If no unlabeled support vectors found, use a fallback strategy, for this case Entropy + Euclidean distance
                    entropies = entropy(probs, axis=1)
                    dists = pairwise_distances(X_pool, X_labeled, metric="euclidean")
                    scaled_dists = dists * entropies[:, None]
                    min_scaled_dist = np.min(scaled_dists, axis=1)
                    next_idx = np.argmax(min_scaled_dist)
            else:
                raise ValueError("The 'support' strategy requires a TSVM model.")


        X_labeled = np.vstack([X_labeled, X_pool[next_idx]])
        y_labeled = np.append(y_labeled, y_pool[next_idx])
        X_pool = np.delete(X_pool, next_idx, axis=0)
        y_pool = np.delete(y_pool, next_idx)
        iteration += 1

    results = {
        "accuracy_list": accuracy_list,
        "auc_list": auc_list,
        "point_mu": point_mu,
        "point_ou": point_ou,
        "point_mee": point_mee,
    }
    return results