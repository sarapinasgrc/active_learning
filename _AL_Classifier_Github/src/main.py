from utils.selection import mean_distance, minimum_distance
from utils.active_loop import al_loop

def active_learning(X_initial, y_initial, X_unlabeled, y_unlabeled, X_test, y_test, n_samples=3, model='tsvm', selection='mean', al='coreset', kernel='linear'):
    """
    Active learning loop with selection initial strategy and active learning strategy.

        Parameters:
        - X_initial: np.array, Initial labeled data.
        - y_initial: np.array, Initial labels.
        - X_unlabeled: np.array, Unlabeled data.
        - y_unlabeled: np.array, Unlabeled labels.
        - X_test: np.array, Test data.
        - y_test: np.array, Test labels.
        - n_samples: int, Number of samples to select at each iteration.
        - model: str, Model to use.
        - selection: str, Initial selection strategy.
        - al: str, Active learning strategy.
        - kernel: str, Kernel to use.

        Returns:
        - results: dict, Results of the active learning loop
    """

    if selection=='minimum':
        X_pool, y_pool, X_labeled, y_labeled =  minimum_distance(X_unlabeled, y_unlabeled, X_initial, y_initial, n_samples=n_samples)

    if selection=='mean':
        X_pool, y_pool, X_labeled, y_labeled =  mean_distance(X_unlabeled, y_unlabeled, X_initial, y_initial, n_samples=n_samples)

    results = al_loop(X_labeled, y_labeled, X_pool, y_pool, X_test, y_test, model=model, al=al, kernel=kernel)

    return results    

def active_leaning_direct(X_initial, y_initial, X_unlabeled, y_unlabeled, X_test, y_test, model='tsvm', al='coreset', kernel='linear'):
  """
  Active learning loop with direct initial strategy.
  
    Parameters:
    - X_initial: np.array, Initial labeled data.
    - y_initial: np.array, Initial labels.
    - X_unlabeled: np.array, Unlabeled data.
    - y_unlabeled: np.array, Unlabeled labels.
    - X_test: np.array, Test data.
    - y_test: np.array, Test labels.
    - model: str, Model to use.
    - al: str, Active learning strategy.
    - kernel: str, Kernel to use.

    Returns:
    - results: dict, Results of the active learning loop

  """

  results = al_loop(X_initial, y_initial, X_unlabeled, y_unlabeled, X_test, y_test, model=model, al=al, kernel=kernel)

  return results