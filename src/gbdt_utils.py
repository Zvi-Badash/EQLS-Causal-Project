import warnings

import numpy as np
import shap
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def _default_metric_fn(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred > 0.5),
        "brier_score": -brier_score_loss(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred > 0.5),
    }


DEFAULT_PARAM_SPACE = {
    "num_leaves": hp.quniform("num_leaves", 20, 150, 1),
    "max_depth": hp.quniform("max_depth", 3, 15, 1),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
    "min_data_in_leaf": hp.quniform("min_data_in_leaf", 20, 100, 1),
    "lambda_l1": hp.loguniform("lambda_l1", np.log(1e-8), np.log(10)),
    "lambda_l2": hp.loguniform("lambda_l2", np.log(1e-8), np.log(10)),
    "feature_fraction": hp.uniform("feature_fraction", 0.6, 1.0),
    "bagging_fraction": hp.uniform("bagging_fraction", 0.6, 1.0),
    "bagging_freq": hp.quniform("bagging_freq", 1, 10, 1),
}


class GBDTClassifier(BaseEstimator, ClassifierMixin):
    """
    LightGBM classifier with SHAP-based feature selection and optional Hyperopt tuning
    using k-fold cross-validation. Supports LightGBM backend, GPU training,
    custom metric functions, and scikit-learn API.

    Parameters
    ----------
    model_params : dict, optional
        Base parameters passed to the underlying LightGBM estimator.
    param_space : dict, optional
        Hyperopt search space for tuning. If None, a balanced default space is used.
    max_evals : int, default=50
        Maximum number of Hyperopt evaluations.
    k_folds : int, default=5
        Number of folds for cross-validation during tuning.
    random_state : int, default=42
        Random seed.
    frac_features_keep : float, default=1.0
        Fraction of top features to keep based on mean absolute SHAP values.
    gpu : bool, default=False
        Whether to use GPU training if available.
    do_tune : bool, default=True
        Whether to perform hyperparameter tuning. If False, skip tuning.
    compute_metric_fn : callable, optional
        Function (y_true, y_pred_proba) -> dict of metrics. If None, accuracy is used.
    metric_key : str, default='loss'
        Key in metric dict to optimize (higher is better).
    """

    def __init__(
        self,
        model_params=None,
        param_space=None,
        max_evals=25,
        k_folds=10,
        random_state=42,
        frac_features_keep=1.0,
        gpu=False,
        do_tune=True,
        compute_metric_fn=None,
        metric_key="accuracy",
    ) -> None:
        self.model_params = model_params
        self.param_space = param_space
        self.max_evals = max_evals
        self.k_folds = k_folds
        self.random_state = random_state
        self.frac_features_keep = frac_features_keep
        self.gpu = gpu
        self.do_tune = do_tune
        self.compute_metric_fn = compute_metric_fn
        self.metric_key = metric_key
        self._frozen = False

    def _create_model(self, additional_params=None) -> lgb.LGBMClassifier:
        base_params = self.model_params or {}
        params = {**base_params}
        if additional_params:
            for k, v in additional_params.items():
                if k in ("num_leaves", "max_depth", "min_data_in_leaf", "bagging_freq"):
                    params[k] = int(v)
                else:
                    params[k] = v
        if self.gpu:
            params.setdefault("device", "gpu")
        if lgb is None:
            raise ImportError("lightgbm is not installed")
        return lgb.LGBMClassifier(
            random_state=self.random_state, verbosity=-1, **params
        )

    def freeze(self):
        """
        Freeze the estimator so that **all future .fit() calls reuse the
        current hyper-parameters** (no Hyperopt search) and so that a cloned
        estimator keeps exactly the same settings.

        Call it *after* an initial fit–tune cycle, or beforehand if you just
        want to skip tuning from the outset.
        """
        # If tuning already ran, bake the winners into `model_params`
        if hasattr(self, "best_params_") and getattr(self, "best_params_", {}):
            merged = {**(self.model_params or {}), **self.best_params_}
            # Store them in an __init__ argument so sklearn.clone() carries them
            for k in ("num_leaves", "max_depth", "min_data_in_leaf", "bagging_freq"):
                merged[k] = int(merged[k])
            self.model_params = merged

        # Switch off the tuner
        self.do_tune = False
        self.param_space = None  # makes search_space falsy
        self._frozen = True
        return self

    def fit(self, X, y):
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        y_arr = np.asarray(y)

        init_model = self._create_model()
        init_model.fit(X_arr, y_arr)

        explainer = shap.TreeExplainer(init_model)
        shap_vals = explainer.shap_values(X_arr)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_imp = np.abs(shap_vals).mean(axis=0)

        total_feats = X_arr.shape[1]
        keep = max(1, int(self.frac_features_keep * total_feats))
        top_idx = np.argsort(mean_imp)[-keep:]
        self.selected_indices_ = np.sort(top_idx)
        X_sel = X_arr[:, self.selected_indices_]

        metric_fn = self.compute_metric_fn or _default_metric_fn
        search_space = self.param_space or DEFAULT_PARAM_SPACE

        if self.do_tune and search_space:

            def objective(params):
                cv = StratifiedKFold(
                    n_splits=self.k_folds, shuffle=True, random_state=self.random_state
                )
                scores = []
                for train_idx, val_idx in cv.split(X_sel, y_arr):
                    m = self._create_model(params)
                    m.fit(X_sel[train_idx], y_arr[train_idx])
                    proba = m.predict_proba(X_sel[val_idx])[:, 1]
                    metrics = metric_fn(y_arr[val_idx], proba)
                    scores.append(metrics[self.metric_key])
                return {"loss": -np.mean(scores), "status": STATUS_OK}

            trials = Trials()
            best = fmin(
                fn=objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals=self.max_evals,
                trials=trials,
                rstate=np.random.default_rng(self.random_state),
            )
            self.best_params_ = best
        else:
            self.best_params_ = {}

        self.best_model_ = self._create_model(self.best_params_)
        self.best_model_.fit(X_sel, y_arr)
        return self

    def predict(self, X):
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        X_sel = X_arr[:, self.selected_indices_]
        return self.best_model_.predict(X_sel)

    def predict_proba(self, X):
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        X_sel = X_arr[:, self.selected_indices_]
        return self.best_model_.predict_proba(X_sel)
