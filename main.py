import torch
from torch.optim.lr_scheduler import LambdaLR
from dataset import get_corrupt_data_loader, get_data_loader
import os
from train import load_checkpoint, evaluate_and_save
from train import save_checkpoint, train_epoch, get_vi_weight
from model import StoResNet18


lr = 1e-4
wd = 1e-5
num_epochs = 100
det_milestones = (0.5, 0.9)
sto_milestones = (0.5, 0.9)
lr_ratio_det = 0.01
lr_ratio_sto = 1/3


# function used by our scheduler
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

    # Entraînement
    trainmem = []
    valmem = []

    save_dir = os.path.join(base_dir, run_id)
    for epoch in range(num_epochs):
        # Entraîne pour une époque
        eloglike, kl, entropy = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, num_train_sample, kl_type, gamma, entropy_type, n_batch
        )

        # Affiche les métriques
        if (epoch + 1) % logging_freq == 0:
            vi_weight = get_vi_weight(epoch, kl_min, kl_max, last_iter)
            lr1 = optimizer.param_groups[0]['lr']
            lr2 = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else 0.0
            print(f"Epoch {epoch}: loglike: {eloglike:.4f}, kl: {kl:.4f}, entropy: {entropy:.4f}, kl weight: {vi_weight:.4f}, lr1: {lr1:.4f}, lr2: {lr2:.4f}")

        # Sauvegarde un checkpoint intermédiaire
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = save_checkpoint(model, save_dir, epoch + 1)
            print(f"Saved checkpoint to {checkpoint_path}")

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
        num_epochs=1,
        logging_freq=1,
        kl_type="mean",  # mean, full, upper_bound
        gamma=1.0,
        entropy_type="conditional",
        det_checkpoint=None,
        dataset="cifar10",
        save_freq=1,
        base_dir="./experiments",
        run_id="run_001",
        model=model,
        scheduler=scheduler,
        optimizer=optimizer
    )
