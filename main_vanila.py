import torch
from torch.optim.lr_scheduler import LambdaLR
from dataset import get_corrupt_data_loader, get_data_loader
import os
from eval import eval
from train_vanila import load_checkpoint, evaluate_and_save
from train_vanila import save_checkpoint, train_epoch, get_vi_weight
from model import StoResNet18
import os
import json
import numpy as np


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


def save_metrics_history(history, save_dir, split_name):
    """
    Sauvegarde l'historique des métriques dans des fichiers JSON
    """
    os.makedirs(save_dir, exist_ok=True)
    for metric_name, values in history.items():
        filepath = os.path.join(save_dir, f"{split_name}_{metric_name}_history_basic.json")
        with open(filepath, 'w') as f:
            json.dump({
                'metric': metric_name,
                'split': split_name,
                'values': values
            }, f, indent=4)
        print(f"Saved {split_name} {metric_name} history to {filepath}")


def schedule(num_epochs, epoch, milestones, lr_ratio):
    t = epoch / num_epochs
    m1, m2 = milestones
    if t <= m1:
        factor = 1.0
    elif t <= m2:
        factor = 1.0 - (1.0 - lr_ratio) * (t - m1) / (m2 - m1)
    else:
        factor = lr_ratio
    return factor


# functions to get data
def get_dataloader(batch_size=8, test_batch_size=8, num_test_sample=1,
                   validation=False, validation_fraction=5000,
                   dataset='cifar10', augment_data=True,
                   num_train_workers=1, num_test_workers=1,
                   data_norm_stat=None):
    test_bs = test_batch_size // num_test_sample
    return get_data_loader(
        dataset, train_bs=batch_size, test_bs=test_bs,
        validation=validation, validation_fraction=validation_fraction,
        augment=augment_data, num_train_workers=num_train_workers,
        num_test_workers=num_test_workers, norm_stat=data_norm_stat
    )


def get_corruptdataloader(intensity=1, test_batch_size=8, num_test_sample=1,
                          dataset='cifar10', num_test_workers=1, data_norm_stat=None):
    test_bs = test_batch_size // num_test_sample
    return get_corrupt_data_loader(
        dataset, intensity, batch_size=test_bs,
        root_dir='data/', num_workers=num_test_workers,
        norm_stat=data_norm_stat
    )


def main(num_train_sample, device, validation, num_epochs, logging_freq,
         kl_type, gamma, entropy_type, det_checkpoint, dataset, save_freq,
         base_dir, run_id, model, scheduler, optimizer):

    kl_min = 0.0
    kl_max = 1.0
    last_iter = 200

    # Charge les données
    if validation:
        train_loader, valid_loader, test_loader = get_dataloader(validation=True)
        print(f"Train size: {len(train_loader.dataset)}, validation size: {len(valid_loader.dataset)}, test size: {len(test_loader.dataset)}")
    else:
        train_loader, test_loader = get_dataloader()
        print(f"Train size: {len(train_loader.dataset)}, test size: {len(test_loader.dataset)}")

    n_batch = len(train_loader)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(model)

    if False:
        model = load_checkpoint(model, det_checkpoint, device)
        print(f"Loaded deterministic checkpoint from {det_checkpoint}")

    save_dir = os.path.join(base_dir, run_id)
    for epoch in range(num_epochs):
        # Entraîne pour une époque
        eloglike, kl, entropy = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            num_train_sample, kl_type, gamma, entropy_type, n_batch
        )

        # Stocke les métriques d'entraînement
        train_history['loglike'].append(float(eloglike))
        train_history['kl'].append(float(kl))
        train_history['entropy'].append(float(entropy))

        # Évalue sur validation et test
        val_metrics = eval(model, valid_loader, device, 1, 15)
        test_metrics = eval(model, test_loader, device, 1, 15)

        # Stocke les métriques de validation
        val_history['nll'].append(val_metrics['nll'])
        val_history['nll_miss'].append(val_metrics['nll_miss'])
        val_history['ece'].append(val_metrics['ece'])
        val_history['predictive_entropy_total_mean'].append(val_metrics['predictive_entropy']['total'][0])
        val_history['predictive_entropy_total_std'].append(val_metrics['predictive_entropy']['total'][1])
        val_history['predictive_entropy_aleatoric_mean'].append(val_metrics['predictive_entropy']['aleatoric'][0])
        val_history['predictive_entropy_aleatoric_std'].append(val_metrics['predictive_entropy']['aleatoric'][1])
        val_history['predictive_entropy_epistemic_mean'].append(val_metrics['predictive_entropy']['epistemic'][0])
        val_history['predictive_entropy_epistemic_std'].append(val_metrics['predictive_entropy']['epistemic'][1])
        val_history['top-1'].append(val_metrics['top-1'])
        val_history['top-2'].append(val_metrics['top-2'])
        val_history['top-3'].append(val_metrics['top-3'])

        # Stocke les métriques de test
        test_history['nll'].append(test_metrics['nll'])
        test_history['nll_miss'].append(test_metrics['nll_miss'])
        test_history['ece'].append(test_metrics['ece'])
        test_history['predictive_entropy_total_mean'].append(test_metrics['predictive_entropy']['total'][0])
        test_history['predictive_entropy_total_std'].append(test_metrics['predictive_entropy']['total'][1])
        test_history['predictive_entropy_aleatoric_mean'].append(test_metrics['predictive_entropy']['aleatoric'][0])
        test_history['predictive_entropy_aleatoric_std'].append(test_metrics['predictive_entropy']['aleatoric'][1])
        test_history['predictive_entropy_epistemic_mean'].append(test_metrics['predictive_entropy']['epistemic'][0])
        test_history['predictive_entropy_epistemic_std'].append(test_metrics['predictive_entropy']['epistemic'][1])
        test_history['top-1'].append(test_metrics['top-1'])
        test_history['top-2'].append(test_metrics['top-2'])
        test_history['top-3'].append(test_metrics['top-3'])

        # Affiche les métriques
        if (epoch + 1) % 1 == 0:
            vi_weight = get_vi_weight(epoch, kl_min, kl_max, last_iter)
            lr1 = optimizer.param_groups[0]['lr']
            lr2 = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else 0.0
            print(f"Epoch {epoch}: loglike: {eloglike:.4f}, kl: {kl:.4f}, entropy: {entropy:.4f}, "
                  f"kl weight: {vi_weight:.4f}, lr1: {lr1:.4f}, lr2: {lr2:.4f}")
            print(f"Validation metrics: NLL: {val_metrics['nll']:.4f}, ECE: {val_metrics['ece']:.4f}, "
                  f"Top-1: {val_metrics['top-1']:.4f}")
            print(f"Test metrics: NLL: {test_metrics['nll']:.4f}, ECE: {test_metrics['ece']:.4f}, "
                  f"Top-1: {test_metrics['top-1']:.4f}")

        # Sauvegarde un checkpoint intermédiaire
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = save_checkpoint(model, save_dir, epoch + 1)
            print(f"Saved checkpoint to {checkpoint_path}")

        metrics_save_dir = os.path.join(save_dir, "metrics_history")
        save_metrics_history(train_history, metrics_save_dir, "train")
        save_metrics_history(val_history, metrics_save_dir, "val")
        save_metrics_history(test_history, metrics_save_dir, "test")

    # Sauvegarde le checkpoint final
    final_checkpoint_path = save_checkpoint(model, save_dir)
    print(f"Saved final checkpoint to {final_checkpoint_path}")

    # Charge le checkpoint final pour l'évaluation
    model = load_checkpoint(model, final_checkpoint_path, device)

    # Évaluation sur l'ensemble de test
    test_result = evaluate_and_save(
        model, test_loader, os.path.join(base_dir, run_id, dataset, 'test_result.json'), device
    )
    print(f"Test results saved: {test_result}")

    # Évaluation sur l'ensemble de validation (si activé)
    if validation:
        valid_result = evaluate_and_save(
            model, valid_loader, os.path.join(base_dir, run_id, dataset, 'valid_result.json'), device
        )
        print(f"Validation results saved: {valid_result}")

    # Évaluation sur les données corrompues
    for intensity in range(5):
        corrupted_loader = get_corruptdataloader(intensity)
        corrupted_result = evaluate_and_save(
            model, corrupted_loader, os.path.join(base_dir, run_id, dataset, str(intensity), 'result.json'), device
        )
        print(f"Corrupted data (intensity {intensity}) results saved: {corrupted_result}")


if __name__ == "__main__":

    sgd_params = {
        'momentum': 0.9,
        'dampening': 0.0,
        'nesterov': True
    }
    det_params = {
        'lr': 0.1, 'weight_decay': 5e-4
    }
    sto_params = {
        'lr': 0.1, 'weight_decay': 0.0, 'momentum': 0.0, 'nesterov': False
    }

    # model, optimizer, scheduler for trainning
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    model = StoResNet18(10, 2, 1., 0.5, (1.0, 0.5), (0.05, 0.02), 0.1, mode='in')
    model = model.to(device)
    optimizer = optimizer = torch.optim.SGD(
        [
            {'params': [p for n, p in model.named_parameters() if 'posterior' not in n and 'prior' not in n], **det_params},
            {'params': [p for n, p in model.named_parameters() if 'posterior' in n or 'prior' in n], **sto_params}
        ], **sgd_params
    )
    scheduler = LambdaLR(optimizer,
                         [lambda e: schedule(
                          num_epochs, e, det_milestones, lr_ratio_det),
                          lambda e: schedule(num_epochs,
                                             e,
                                             sto_milestones,
                                             lr_ratio_sto
                                             )]
                         )
    main(
        num_train_sample=1,
        device=device,
        validation=True,
        num_epochs=10,
        logging_freq=1,
        kl_type="mean",  # mean, full, upper_bound
        gamma=1.0,
        entropy_type="conditional",
        det_checkpoint=None,
        dataset="cifar10",
        save_freq=1,
        base_dir="./experiments",
        run_id="run_basic_loss",
        model=model,
        scheduler=scheduler,
        optimizer=optimizer
    )
