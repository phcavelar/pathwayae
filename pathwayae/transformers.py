import warnings
import logging

import numpy as np

import sklearn.base
import sklearn.utils
import sklearn.utils.validation

logger = logging.getLogger()

class ClipTransformer(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    """
    A transformer that either clips data according to predefined minima and maxima or learns the minima and maxima present in the training set to be the input limits in other sets.
    This transformer's transform has no inverse.
    """

    def __init__(self, *, min=None, max=None, from_data=False, min_axis=None, max_axis=None):
        self.from_data = from_data
        if min is not None and max is not None:
            self.min = min
            self.max = max
        elif not from_data and (min is not None or max is not None):
            raise ValueError("Either both min and max must not be none, or you must train from data")
        self.min_axis = min_axis
        self.max_axis = max_axis
    
    def _reset(self):
        if hasattr(self, "min"):
            del self.min
            del self.max

    def fit(self, X, y=None, sample_weight=None):
        if self.from_data:
            self._reset()
            self.min = np.min(X, axis=self.min_axis)
            self.max = np.max(X, axis=self.max_axis)
        return self
    
    def transform(self, X, copy=None):
        sklearn.utils.validation.check_is_fitted(self, ["min","max"])
        new_x:np.ndarray = np.clip(X, a_min=self.min, a_max=self.max)
        logger.debug(self.min, new_x.min(), self.max, new_x.max())
        return new_x