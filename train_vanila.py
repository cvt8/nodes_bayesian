import os
import json
import torch
from eval import eval
from tqdm.rich import tqdm


def get_vi_weight(epoch, kl_min, kl_max, last_iter):
    value = (kl_max-kl_min)/last_iter
    return min(kl_max, kl_min + epoch*value)


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch,
                num_train_sample, kl_type, gamma, entropy_type, n_batch):
    model.train()
    total_eloglike = 0.0
    last_kl = 0.0
    last_entropy = 0.0
    kl_min = 0.0
    kl_max = 1.0
    last_iter = 200
    lambda_ = 1
    for inputs, targets in tqdm(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # First pass with gradient calculation
        optimizer.zero_grad()
        eloglike, kl, entropy, logits, alpha = model.vi_loss(
            inputs, targets, num_train_sample, kl_type, entropy_type
        )
        vi_weight = get_vi_weight(epoch, kl_min, kl_max, last_iter)
        # mutual_info = model.mutual_information(inputs, logits, alpha, num_train_sample, entropy_type)
        loss = eloglike - vi_weight * (kl - gamma * entropy) / (n_batch * inputs.size(0))
        # elbo_loss = eloglike - vi_weight * (kl ) / (n_batch * inputs.size(0))

        loss.backward()
        optimizer.step()

        # Save the loss for comparison
        # loss_first_pass = elbo_loss.detach().item()

        # Second pass without gradient calculation
        # with torch.no_grad():
        #     eloglike, kl, entropy, logits, alpha = model.vi_loss(
        #         inputs, targets, num_train_sample, kl_type, entropy_type
        #     )
        #     vi_weight = get_vi_weight(epoch, kl_min, kl_max, last_iter)
        #     mutual_info = model.mutual_information(inputs, logits, alpha, num_train_sample, entropy_type)
        #     loss = eloglike - vi_weight * (kl - gamma * entropy + lambda_ * mutual_info) / (n_batch * inputs.size(0))
        #     elbo_loss = eloglike - vi_weight * (kl ) / (n_batch * inputs.size(0))

        # # Calculate the difference in loss
        # loss_second_pass = elbo_loss.item()
        # loss_diff = loss_first_pass - loss_second_pass
        # print(loss_diff)

        # Update alpha based on the sign of the loss difference
        # model.update_alpha(logits, loss_diff) #A vérifier

        total_eloglike += eloglike.detach().item()
        last_kl = kl.item()
        last_entropy = entropy.item()
        # last_mutual_info = mutual_info.item()

    scheduler.step()
    total_eloglike /= len(train_loader)

    return total_eloglike, last_kl, last_entropy


def save_checkpoint(model, save_dir, epoch=None):
    os.makedirs(save_dir, exist_ok=True)
    if epoch is None:
        checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')
    else:
        checkpoint_path = os.path.join(save_dir, f'checkpoint{epoch}.pt')
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def load_checkpoint(model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def evaluate_and_save(model, dataloader, save_path, device):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = eval(model, dataloader, device, 1, 15)
    with open(save_path, 'w') as f:
        json.dump(results, f)
    return results
