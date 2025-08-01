import sys
import os
import glob
import shutil
import re
import torch
from torch.optim.lr_scheduler import LambdaLR
import hydra
from omegaconf import DictConfig

from model import StoResNet18
from main import schedule
# reuse main function but with configurable lambda_info and gamma
from main import main as run_main


def find_highest_checkpoint(checkpoint_dir):
    """Find the highest numbered checkpoint in a directory and its subdirectories."""
    if not os.path.exists(checkpoint_dir):
        return -1, -1
    
    # Search recursively for checkpoint files
    checkpoint_files = []
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.startswith("checkpoint") and file.endswith(".pt"):
                checkpoint_files.append(os.path.join(root, file))
    
    if not checkpoint_files:
        return -1, -1
    
    # Extract epoch numbers from checkpoint filenames
    highest_epoch = -1
    highest_checkpoint = -1
    
    for checkpoint_file in checkpoint_files:
        filename = os.path.basename(checkpoint_file)
        # Match checkpoint followed by optional number and .pt
        match = re.match(r'checkpoint(\d*)\.pt', filename)
        if match:
            epoch_str = match.group(1)
            if epoch_str == '':
                # checkpoint.pt without number, assume epoch 0
                epoch = 0
            else:
                epoch = int(epoch_str)
            
            if epoch > highest_epoch:
                highest_epoch = epoch
                highest_checkpoint = checkpoint_file
    
    return highest_checkpoint, highest_epoch


def copy_metrics_history(source_dir, target_dir):
    """Copy metrics_history directory from source to target."""
    source_metrics = os.path.join(source_dir, "hydra_experiments/metrics_history")
    target_metrics = os.path.join(target_dir, "hydra_experiments/metrics_history")
    
    if os.path.exists(source_metrics) and not os.path.exists(target_metrics):
        print(f"Copying metrics_history from {source_metrics} to {target_metrics}")
        shutil.copytree(source_metrics, target_metrics)


def find_checkpoint_and_handle_cifar10(cfg):
    """
    Find the appropriate checkpoint and handle CIFAR10 specific logic.
    Returns the checkpoint path or None.
    """
    base_dir = cfg.base_dir
    current_run_id = cfg.run_id
    dataset = cfg.dataset.lower()
    
    print(f"Looking for checkpoints...")
    print(f"Base dir: {base_dir}")
    print(f"Current run ID: {current_run_id}")
    print(f"Dataset: {dataset}")
    
    # First, check for checkpoint in the current run directory
    current_run_dir = os.path.join(base_dir, current_run_id)
    print(f"Checking current run directory: {current_run_dir}")
    current_checkpoint, current_checkpoint_epoch = find_highest_checkpoint(current_run_dir)

    if current_checkpoint_epoch == -1:
        print(f"No checkpoints found in current run directory: {current_run_dir}")
        current_checkpoint = None
        current_checkpoint_epoch = -1

    if current_checkpoint:
        print(f"Found checkpoint in current run directory: {current_checkpoint}")
    
    # For CIFAR10, also check the non-dataset-specific directory
    if dataset == 'cifar10':
        # Extract gamma and lambda values to construct the alternative run_id
        gamma = cfg.gamma
        lambda_info = cfg.lambda_info
        alt_run_id = f"gamma_{gamma}_lambda_{lambda_info}"
        alt_run_dirname = f"gamma_{gamma}_lambda_{lambda_info}"
        alt_run_dir = os.path.join(base_dir, alt_run_dirname)
        print(f"Checking alternative directory for CIFAR10: {alt_run_dir}")

        # Check if the alternative directory exists
        if not os.path.exists(alt_run_dir):
            print(f"Alternative directory {alt_run_dir} does not exist")
            alt_checkpoint = None
            alt_checkpoint_epoch = -1
        else:
            alt_checkpoint, alt_checkpoint_epoch = find_highest_checkpoint(alt_run_dir)

        if alt_checkpoint:
            print(f"Found checkpoint in alternative directory for CIFAR10: {alt_checkpoint}")
            

            if alt_checkpoint_epoch > current_checkpoint_epoch:
                print(f"Using checkpoint from alternative directory: {alt_checkpoint}")
                # Create the current run directory if it doesn't exist
                os.makedirs(current_run_dir, exist_ok=True)
            
                # Copy metrics_history from the alternative directory to current directory
                copy_metrics_history(alt_run_dir, current_run_dir)

                # Copy the checkpoint from alternative directory to current directory
                checkpoint_filename = os.path.basename(alt_checkpoint)
                target_checkpoint_path = os.path.join(current_run_dir, checkpoint_filename)
                print(f"Copying checkpoint from {alt_checkpoint} to {target_checkpoint_path}")
                shutil.copy2(alt_checkpoint, target_checkpoint_path)

                return alt_checkpoint
            else:
                print(f"Current checkpoint is newer or equal to alternative checkpoint, using current checkpoint")
                return current_checkpoint
            
        else:
            print(f"No checkpoint found in alternative directory")
            if current_checkpoint:
                return current_checkpoint

    else:
        if current_checkpoint:
            return current_checkpoint
        
    
    print("No checkpoint found anywhere")
    return None



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

# Handle optional --dataset argument before Hydra parses CLI
if "--dataset" in sys.argv:
    idx = sys.argv.index("--dataset")
    if idx + 1 < len(sys.argv):
        dataset = sys.argv[idx + 1]
        # Remove custom arguments so Hydra doesn't error on unknown args
        del sys.argv[idx:idx + 2]
        sys.argv.append(f"dataset={dataset}")

        

@hydra.main(version_base=None, config_path="configs", config_name="config")
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

    # Find the appropriate checkpoint using the new logic
    det_checkpoint = find_checkpoint_and_handle_cifar10(cfg)
    if det_checkpoint:
        print(f"Using checkpoint: {det_checkpoint}")
    else:
        print("No checkpoint found, starting from scratch")
        det_checkpoint = cfg.det_checkpoint if "det_checkpoint" in cfg else None

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