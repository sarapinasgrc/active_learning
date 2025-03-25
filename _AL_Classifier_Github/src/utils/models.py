import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _num_samples
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from LAMDA_SSL.Algorithm.Classification.TSVM import TSVM

class MultiClassTSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel, **tsvm_params):
        self.kernel = kernel
        self.tsvm_params = tsvm_params
        self.classifiers = {}

    def fit(self, X_initial, y_initial, X_pool):
        self.classes_ = np.unique(y_initial)
        self.classifiers = {}

        for cls in self.classes_:
            y_binary = np.where(y_initial == cls, 1, -1)

            clf = TSVM(kernel=self.kernel, **self.tsvm_params)
            clf.fit(X_initial, y_binary, X_pool)
            self.classifiers[cls] = clf

    def predict(self, X): 

        check_is_fitted(self)

        n_samples = _num_samples(X)
        maxima = np.empty(n_samples, dtype=float)
        maxima.fill(-np.inf)
        argmaxima = np.zeros(n_samples, dtype=int)
        for i, (_, clf) in enumerate(self.classifiers.items()):
            pred = clf.base_estimator.decision_function(X)
            np.maximum(maxima, pred, out=maxima)
            argmaxima[maxima == pred] = i

        return np.array(list(self.classifiers.keys()))[argmaxima]

    def predict_proba(self, X):
        check_is_fitted(self)
        probas = np.array([clf.predict_proba(X, Transductive=False)[:, 1] for clf in self.classifiers.values()]).T

        if len(self.classifiers) == 1:
            probas = np.concatenate(((1 - probas), probas), axis=1)

        probas /= np.sum(probas, axis=1)[:, np.newaxis]
        return probas



def define_model(model, kernel):
  if model == 'svm':
        model = SVC(kernel=kernel, probability=True, random_state=42)
  elif model == 'gp':
      kernel = DotProduct() + WhiteKernel()
      model = GaussianProcessClassifier(kernel=kernel, random_state=42)
  elif model == 'logreg':
      model = LogisticRegression(random_state=42)
  elif model == 'rf':
      model = RandomForestClassifier(random_state=42)
  elif model == 'multiclasstsvm':
      model = MultiClassTSVM(kernel=kernel)  
  elif model == 'tsvm':
      model = TSVM(kernel=kernel)
  return model