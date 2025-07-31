import sys
import torch
from torch.optim.lr_scheduler import LambdaLR
import hydra
from omegaconf import DictConfig

from model import StoResNet18
from main import schedule
# reuse main function but with configurable lambda_info and gamma
from main import main as run_main



train_history = {
    'loglike': [],
    'kl': [],
    'entropy': []
}

val_history = {
    'nll': [],
    'nll_miss': [],
    'ece': [],
    'predictive_entropy_total_mean': [],
    'predictive_entropy_total_std': [],
    'predictive_entropy_aleatoric_mean': [],
    'predictive_entropy_aleatoric_std': [],
    'predictive_entropy_epistemic_mean': [],
    'predictive_entropy_epistemic_std': [],
    'top-1': [],
    'top-2': [],
    'top-3': []
}

test_history = {
    'nll': [],
    'nll_miss': [],
    'ece': [],
    'predictive_entropy_total_mean': [],
    'predictive_entropy_total_std': [],
    'predictive_entropy_aleatoric_mean': [],
    'predictive_entropy_aleatoric_std': [],
    'predictive_entropy_epistemic_mean': [],
    'predictive_entropy_epistemic_std': [],
    'top-1': [],
    'top-2': [],
    'top-3': []
}

lr = 1e-4
wd = 1e-5
num_epochs = 100
det_milestones = (0.5, 0.9)
sto_milestones = (0.5, 0.9)
lr_ratio_det = 0.01
lr_ratio_sto = 1/3



# Handle optional --parallel-workers argument before Hydra parses CLI
if "--parallel-workers" in sys.argv:
    idx = sys.argv.index("--parallel-workers")
    if idx + 1 < len(sys.argv):
        workers = sys.argv[idx + 1]
        # Remove custom arguments so Hydra doesn't error on unknown args
        del sys.argv[idx:idx + 2]
        # Append hydra overrides to enable joblib launcher
        sys.argv.append("hydra/launcher=joblib")
        sys.argv.append(f"hydra.launcher.n_jobs={workers}")

@hydra.main(config_path="configs", config_name="config")
def run(cfg: DictConfig) -> None:
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    import multiprocessing
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    
    sgd_params = {"momentum": 0.9, "dampening": 0.0, "nesterov": True}
    det_params = {"lr": 0.1, "weight_decay": 5e-4}
    sto_params = {"lr": 0.1, "weight_decay": 0.0, "momentum": 0.0, "nesterov": False}

    det_milestones = (0.5, 0.9)
    sto_milestones = (0.5, 0.9)
    lr_ratio_det = 0.01
    lr_ratio_sto = 1 / 3

    # Force CPU for multiprocessing compatibility, or handle device creation inside run_main
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model and move to device inside this process
    model = StoResNet18(10, 2, 1.0, 0.5, (1.0, 0.5), (0.05, 0.02), 0.1, mode="in")
    model = model.to(device)
    
    optimizer = torch.optim.SGD(
        [
            {"params": [p for n, p in model.named_parameters() if "posterior" not in n and "prior" not in n], **det_params},
            {"params": [p for n, p in model.named_parameters() if "posterior" in n or "prior" in n], **sto_params},
        ],
        **sgd_params,
    )
    scheduler = LambdaLR(
        optimizer,
        [
            lambda e: schedule(cfg.num_epochs, e, det_milestones, lr_ratio_det),
            lambda e: schedule(cfg.num_epochs, e, sto_milestones, lr_ratio_sto),
        ],
    )

    det_checkpoint = cfg.det_checkpoint if "det_checkpoint" in cfg else None #take the latest checkpoint if not specified

    run_main(
        num_train_sample=cfg.num_train_sample,
        device=device,
        validation=cfg.validation,
        num_epochs=cfg.num_epochs,
        logging_freq=1,
        kl_type=cfg.kl_type,
        gamma=cfg.gamma,
        lambda_info=cfg.lambda_info,
        entropy_type=cfg.entropy_type,
        det_checkpoint=det_checkpoint,
        dataset=cfg.dataset,
        save_freq=1,
        base_dir=cfg.base_dir,
        run_id=cfg.run_id,
        model=model,
        scheduler=scheduler,
        optimizer=optimizer,
    )

if __name__ == "__main__":
    # Set torch multiprocessing sharing strategy
    torch.multiprocessing.set_sharing_strategy('file_system')
    run()