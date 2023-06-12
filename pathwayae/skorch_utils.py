import warnings

import numpy as np

import skorch
import skorch.scoring
import skorch.exceptions
import skorch.utils

class ScoredNeuralNetRegressor(skorch.NeuralNetRegressor):
    def score(self, X, y=None, sample_weight=None):
        return -skorch.scoring.loss_scoring(self, X, y, sample_weight) # Negative so that sklearn gets the smallest loss

class ScoredNeuralNetAutoencoder(skorch.NeuralNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def score(self, X, y=None, sample_weight=None):
        return -skorch.scoring.loss_scoring(self, X, X, sample_weight) # Negative so that sklearn gets the smallest loss
    
    def transform(self, X):
        return self.predict(X)

    def full_transform(self, X):
        nonlin = self._get_predict_nonlinearity()
        ys:list[list[np.ndarray]] = None
        for yp in self.forward_iter(X, training=False):
            if not isinstance(yp, tuple):
                yp = (yp,)
            if ys is None:
                ys = [[] for _ in yp]
            for i, _ in enumerate(yp):
                ys[i].append(skorch.utils.to_numpy(nonlin(yp[i])))
        ret = [np.concatenate(ys[i], 0) for i, _ in enumerate(yp)]
        return tuple(ret)

    
    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y, **kwargs)
        Xt = self.transform(X)
        return Xt
    
    def fit_loop(self, X, y=None, epochs=None, **fit_params):
        super().fit_loop.__doc__
        return super().fit_loop(X, y=X, epochs=epochs, **fit_params)


class EarlyStoppingWithScoreToStartAfter(skorch.callbacks.EarlyStopping):
    def __init__(
            self,
            monitor='valid_loss',
            patience=5,
            threshold=1e-4,
            threshold_mode='rel',
            lower_is_better=True,
            sink=print,
            score_to_start_after=None,
            score_to_start_after_monitor = "train_loss",
            ):
        super(EarlyStoppingWithScoreToStartAfter,self).__init__(
            monitor=monitor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            lower_is_better=lower_is_better,
            sink=sink,
            )
        if score_to_start_after is None:
            warnings.warn("minimal_score was not defined, this class will act as EarlyStopping.")
        self.score_to_start_after = score_to_start_after
        self.score_to_start_after_monitor = score_to_start_after_monitor
    
    def _check_passed_score_to_start_after(self, current_score):
        if self.score_to_start_after is None:
            self.passed_score_to_start_after |= True
        elif self.lower_is_better:
            self.passed_score_to_start_after |= current_score < self.score_to_start_after
        else:
            self.passed_score_to_start_after |= current_score > self.score_to_start_after
        return self.passed_score_to_start_after

    def on_train_begin(self, net, **kwargs):
        self.passed_score_to_start_after = False
        super(EarlyStoppingWithScoreToStartAfter,self).on_train_begin(net, **kwargs)

    def on_epoch_end(self, net, **kwargs):
        current_score = net.history[-1, self.score_to_start_after_monitor]
        if self.passed_score_to_start_after or self._check_passed_score_to_start_after(current_score):
            super(EarlyStoppingWithScoreToStartAfter,self).on_epoch_end(net, **kwargs)