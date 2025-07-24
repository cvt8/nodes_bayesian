import torch
from torch.optim.lr_scheduler import LambdaLR
import hydra
from omegaconf import DictConfig

from model import StoResNet18
from main import schedule

# reuse main function but with configurable lambda_info and gamma
from main import main as run_main

@hydra.main(config_path="configs", config_name="config")
def run(cfg: DictConfig) -> None:
    sgd_params = {'momentum': 0.9, 'dampening': 0.0, 'nesterov': True}
    det_params = {'lr': 0.1, 'weight_decay': 5e-4}
    sto_params = {'lr': 0.1, 'weight_decay': 0.0, 'momentum': 0.0, 'nesterov': False}

    det_milestones = (0.5, 0.9)
    sto_milestones = (0.5, 0.9)
    lr_ratio_det = 0.01
    lr_ratio_sto = 1/3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = StoResNet18(10, 2, 1., 0.5, (1.0, 0.5), (0.05, 0.02), 0.1, mode='in')
    model = model.to(device)
    optimizer = torch.optim.SGD([
        {'params': [p for n, p in model.named_parameters() if 'posterior' not in n and 'prior' not in n], **det_params},
        {'params': [p for n, p in model.named_parameters() if 'posterior' in n or 'prior' in n], **sto_params}
    ], **sgd_params)
    scheduler = LambdaLR(optimizer, [
        lambda e: schedule(cfg.num_epochs, e, det_milestones, lr_ratio_det),
        lambda e: schedule(cfg.num_epochs, e, sto_milestones, lr_ratio_sto)
    ])

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
        det_checkpoint=None,
        dataset=cfg.dataset,
        save_freq=1,
        base_dir=cfg.base_dir,
        run_id=cfg.run_id,
        model=model,
        scheduler=scheduler,
        optimizer=optimizer,
    )

if __name__ == "__main__":
    run()
