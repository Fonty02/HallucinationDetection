"""ReducedNonLinear: Autoencoder → alignment in latent space → MLPProber."""

import time

import torch

from ..config import DEVICE, SEED, REDUCED_NONLINEAR_CONFIG
from .training import (
    compute_metrics, count_params, split_train_val,
    train_alignment_network, train_autoencoder, train_mlp_prober,
)


def run_reduced_nonlinear(shared_data: dict, config: dict = None, save_dir: str = None) -> dict:
    """
    1. Train autoencoder for trainer and tester separately.
    2. Encode alignment data + train alignment in latent space.
    3. Train MLPProber on trainer latent.
    4. Project tester latent through alignment, evaluate.
    """
    cfg = config or REDUCED_NONLINEAR_CONFIG
    trainer = shared_data["trainer"]
    tester = shared_data["tester"]
    alignment = shared_data["alignment"]

    # 85/15 split for val inside trainer/tester train sets (autoencoders)
    ae_tr_t, ae_val_t = split_train_val(len(trainer["X_train"]), seed=SEED)
    ae_tr_s, ae_val_s = split_train_val(len(tester["X_train"]), seed=SEED)

    # Shared prober split (fixed across methods)
    prober_split = shared_data["prober_split"]
    pr_tr_idx = prober_split["train_idx"]
    pr_val_idx = prober_split["val_idx"]

    # 1. Train autoencoders (timed separately)
    t0_ae_trainer = time.time()
    ae_trainer, ae_trainer_info = train_autoencoder(
        trainer["X_train"][ae_tr_t], trainer["X_train"][ae_val_t],
        trainer["X_train"].shape[1], cfg,
    )
    ae_trainer_time = time.time() - t0_ae_trainer

    t0_ae_tester = time.time()
    ae_tester, ae_tester_info = train_autoencoder(
        tester["X_train"][ae_tr_s], tester["X_train"][ae_val_s],
        tester["X_train"].shape[1], cfg,
    )
    ae_tester_time = time.time() - t0_ae_tester

    ae_trainer_params = count_params(ae_trainer)
    ae_tester_params = count_params(ae_tester)

    ae_trainer.eval()
    ae_tester.eval()

    # 2. Encode alignment data
    with torch.no_grad():
        z_align_trainer_train = ae_trainer.encode(torch.tensor(alignment["X_trainer_train"], dtype=torch.float32, device=DEVICE)).cpu().numpy()
        z_align_trainer_val = ae_trainer.encode(torch.tensor(alignment["X_trainer_val"], dtype=torch.float32, device=DEVICE)).cpu().numpy()
        z_align_tester_train = ae_tester.encode(torch.tensor(alignment["X_tester_train"], dtype=torch.float32, device=DEVICE)).cpu().numpy()
        z_align_tester_val = ae_tester.encode(torch.tensor(alignment["X_tester_val"], dtype=torch.float32, device=DEVICE)).cpu().numpy()

    # 3. Train alignment in latent space (tester_latent → trainer_latent)
    t0_aligner = time.time()
    align_model, align_info = train_alignment_network(
        z_align_tester_train, z_align_trainer_train,
        z_align_tester_val, z_align_trainer_val, cfg,
    )
    aligner_time = time.time() - t0_aligner

    aligner_params = count_params(align_model)
    aligner_train_n = int(z_align_tester_train.shape[0])

    # 4. Encode trainer data & train prober in latent space
    with torch.no_grad():
        z_trainer_train = ae_trainer.encode(torch.tensor(trainer["X_train"], dtype=torch.float32, device=DEVICE)).cpu().numpy()
        z_trainer_test = ae_trainer.encode(torch.tensor(trainer["X_test"], dtype=torch.float32, device=DEVICE)).cpu().numpy()

    t0_detector = time.time()
    prober, prober_info = train_mlp_prober(
        z_trainer_train[pr_tr_idx], trainer["y_train"][pr_tr_idx],
        z_trainer_train[pr_val_idx], trainer["y_train"][pr_val_idx],
        input_dim=cfg["autoencoder_latent_dim"], cfg=cfg,
    )
    detector_time = time.time() - t0_detector

    detector_params = count_params(prober)
    detector_train_n = int(len(pr_tr_idx))

    # 5. Evaluate trainer
    prober.eval()
    with torch.no_grad():
        zt = torch.tensor(z_trainer_test, dtype=torch.float32, device=DEVICE)
        pred_t = prober.predict(zt).cpu().numpy()
        proba_t = torch.sigmoid(prober(zt)).cpu().numpy()
    metrics_trainer = compute_metrics(trainer["y_test"], pred_t, proba_t)

    # 6. Encode tester test, align, evaluate
    # Notebook uses the model-scaled test data (not the alignment scaler).
    with torch.no_grad():
        z_tester_test = ae_tester.encode(torch.tensor(tester["X_test"], dtype=torch.float32, device=DEVICE))
        z_tester_aligned = align_model(z_tester_test).cpu().numpy()
        zs = torch.tensor(z_tester_aligned, dtype=torch.float32, device=DEVICE)
        pred_s = prober.predict(zs).cpu().numpy()
        proba_s = torch.sigmoid(prober(zs)).cpu().numpy()
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
            "ae_trainer_params": ae_trainer_params,
            "ae_trainer_time_s": ae_trainer_time,
            "ae_tester_params": ae_tester_params,
            "ae_tester_time_s": ae_tester_time,
        },
    }
