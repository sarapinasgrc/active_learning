import numpy as np

def MU_stopping(probs, threshold):
    uncertainty = 1 - np.max(probs, axis=1)
    if np.max(uncertainty) < threshold:
        return True
    return False

def OU_stopping(probs, threshold):
    uncertainty = 1 - np.max(probs, axis=1)
    if np.mean(uncertainty) < threshold:
        return True 
    return False

def MEE_stopping(probs, X_unlabeled, threshold):
    #aux = (1 / len(X_unlabeled)) * sum(1 - max(probs(x)) for x in X_unlabeled)
    aux = np.mean(1 - np.max(probs, axis=1)) 
    if aux < threshold:
        return True
    return False
