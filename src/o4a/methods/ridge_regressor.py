"""RidgeRegressor method: LogisticRegression prober + Ridge alignment."""
import time

from sklearn.linear_model import LogisticRegression, Ridge

from ..config import SEED, RIDGE_REGRESSOR_CONFIG
from .training import compute_metrics, count_params


def run_ridge_regressor(shared_data: dict, config: dict = None, save_dir: str = None) -> dict:
    """
    1. Train LogisticRegression on trainer scaled data.
    2. Evaluate on trainer test (→ trainer metrics).
    3. Train Ridge regressor on concordant alignment data (tester→trainer).
    4. Project tester test through Ridge, evaluate with trainer prober (→ tester metrics).
    """
    cfg = config or RIDGE_REGRESSOR_CONFIG

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

    # 3. Ridge alignment: tester → trainer space
    aligner = Ridge(alpha=cfg["ridge_alpha"], fit_intercept=False)
    t0_aligner = time.time()
    aligner.fit(alignment["X_tester_train"], alignment["X_trainer_train"])
    aligner_time = time.time() - t0_aligner

    aligner_params = count_params(aligner)
    aligner_train_n = int(alignment["X_tester_train"].shape[0])

    # 4. Project tester test & evaluate
    X_tester_scaled = alignment["scaler_tester"].transform(tester["X_test_raw"])
    X_tester_proj = aligner.predict(X_tester_scaled)
    pred_s = clf.predict(X_tester_proj)
    proba_s = clf.predict_proba(X_tester_proj)[:, 1]
    metrics_tester = compute_metrics(tester["y_test"], pred_s, proba_s)

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
