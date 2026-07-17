"""CCA method: LogisticRegression prober + CCA alignment."""
import os
import time

from joblib import dump
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression, LogisticRegression

from ..config import SEED, CCA_CONFIG
from .training import compute_metrics, count_params


class CCAAligner:
    """Wrapper that maps X -> Y via CCA scores + linear regression."""

    def __init__(self, n_components: int, max_iter: int, tol: float, scale: bool):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.scale = scale
        self.cca = CCA(
            n_components=n_components,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
        )
        self.regressor = LinearRegression()
        self.n_features_in_ = None

    def fit(self, X, Y):
        self.cca.fit(X, Y)
        X_scores = self._x_scores(X)
        self.regressor.fit(X_scores, Y)
        self.n_features_in_ = int(X.shape[1])
        return self

    def predict(self, X):
        X_scores = self._x_scores(X)
        return self.regressor.predict(X_scores)

    def get_params(self, deep: bool = True):
        return {
            "n_components": self.n_components,
            "cca_max_iter": self.max_iter,
            "cca_tol": self.tol,
            "cca_scale": self.scale,
        }

    def _x_scores(self, X):
        scores = self.cca.transform(X)
        if isinstance(scores, tuple):
            return scores[0]
        return scores


def run_cca(shared_data: dict, config: dict = None, save_dir: str = None) -> dict:
    """
    1. Train LogisticRegression on trainer scaled data.
    2. Evaluate on trainer test (-> trainer metrics).
    3. Train CCA alignment on concordant data (tester->trainer).
    4. Project tester test through CCA, evaluate with trainer prober (-> tester metrics).
    """
    cfg = config or CCA_CONFIG

    trainer = shared_data["trainer"]
    tester = shared_data["tester"]
    alignment = shared_data["alignment"]

    # 1. Trainer prober (trained on all balanced training data)
    clf = LogisticRegression(
        max_iter=cfg["probe_max_iter"],
        class_weight="balanced",
        solver=cfg["probe_solver"],
        n_jobs=-1,
        random_state=SEED,
    )
    t0_detector = time.time()
    clf.fit(trainer["X_train"], trainer["y_train"])
    detector_time = time.time() - t0_detector

    detector_params = count_params(clf)
    detector_train_n = int(len(trainer["y_train"]))

    # 2. Trainer eval
    pred_t = clf.predict(trainer["X_test"])
    proba_t = clf.predict_proba(trainer["X_test"])[:, 1]
    metrics_trainer = compute_metrics(trainer["y_test"], pred_t, proba_t)

    # 3. CCA alignment: tester -> trainer space
    X_align = alignment["X_tester_train"]
    Y_align = alignment["X_trainer_train"]
    n_components = min(
        cfg["cca_components"],
        X_align.shape[0] - 1,
        X_align.shape[1],
        Y_align.shape[1],
    )
    if n_components < 1:
        raise ValueError("Not enough samples/features for CCA alignment.")

    t0_aligner = time.time()
    aligner = CCAAligner(
        n_components=int(n_components),
        max_iter=cfg["cca_max_iter"],
        tol=cfg["cca_tol"],
        scale=cfg["cca_scale"],
    ).fit(X_align, Y_align)
    aligner_time = time.time() - t0_aligner

    aligner_params = count_params(aligner)
    aligner_train_n = int(X_align.shape[0])

    # 4. Project tester test & evaluate
    X_tester_scaled = alignment["scaler_tester"].transform(tester["X_test_raw"])
    X_tester_proj = aligner.predict(X_tester_scaled)
    pred_s = clf.predict(X_tester_proj)
    proba_s = clf.predict_proba(X_tester_proj)[:, 1]
    metrics_tester = compute_metrics(tester["y_test"], pred_s, proba_s)

    # Save models
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        dump(clf, os.path.join(save_dir, "detector.joblib"))
        dump(aligner, os.path.join(save_dir, "aligner.joblib"))

    return {
        "trainer": metrics_trainer,
        "tester": metrics_tester,
        "_meta": {
            "detector_params": detector_params,
            "detector_time_s": detector_time,
            "detector_train_n": detector_train_n,
            "aligner_params": aligner_params,
            "aligner_time_s": aligner_time,
            "aligner_train_n": aligner_train_n,
        },
    }
