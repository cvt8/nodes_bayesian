import torch
from loss import ECELoss
import numpy as np
from scipy.stats import entropy
import os
import json
from tqdm.rich import tqdm


def eval(model, dataloader, device, num_test_sample, ece_bins):
    model.eval()
    total_samples = len(dataloader.dataset)

    total_nll = 0.0
    nll_miss = 0.0
    top_k_accuracy = [0, 0, 0]
    y_prob_list = []
    y_true_list = []
    y_prob_all_list = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            batch_size = inputs.size(0)
            indices = torch.empty(
                batch_size * num_test_sample, dtype=torch.long, device=device
            )

            prob_list = []
            for idx in range(model.n_components):
                indices.fill_(idx)
                prob = model.forward(inputs, num_test_sample, indices=indices)
                prob_list.append(prob)
            prob = torch.cat(prob_list, dim=1)

            targets_expanded = targets.unsqueeze(1).expand(-1, num_test_sample * model.n_components)

            log_probs = torch.distributions.Categorical(logits=prob).log_prob(targets_expanded)
            log_probs = torch.logsumexp(log_probs, dim=1) - torch.log(torch.tensor(
                num_test_sample * model.n_components, dtype=torch.float32, device=device))
            total_nll -= log_probs.sum().item()

            vote = prob.exp().mean(dim=1)
            top3_preds = torch.topk(vote, k=3, dim=1)[1]

            y_prob_all_list.append(prob.exp().cpu().numpy())
            y_prob_list.append(vote.cpu().numpy())
            y_true_list.append(targets.cpu().numpy())

            incorrect_mask = top3_preds[:, 0] != targets
            if incorrect_mask.sum().item() > 0:
                nll_miss -= log_probs[incorrect_mask].sum().item()

            for k in range(3):
                top_k_accuracy[k] += (top3_preds[:, k] == targets).sum().item()

    total_nll /= total_samples
    nll_miss /= (total_samples - top_k_accuracy[0])
    top_k_accuracy = [acc / total_samples for acc in top_k_accuracy]
    top_k_accuracy = np.cumsum(top_k_accuracy)

    y_prob = np.concatenate(y_prob_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)
    y_prob_all = np.concatenate(y_prob_all_list, axis=0)

    total_entropy = entropy(y_prob, axis=1)
    aleatoric_entropy = entropy(y_prob_all, axis=-1).mean(axis=-1)
    epistemic_entropy = total_entropy - aleatoric_entropy

    ece_loss = ECELoss(n_bins=ece_bins)
    ece_value = ece_loss(torch.from_numpy(y_prob), torch.from_numpy(y_true)).item()

    results = {
        'nll': float(total_nll),
        'nll_miss': float(nll_miss),
        'ece': float(ece_value),
        'predictive_entropy': {
            'total': (float(total_entropy.mean()), float(total_entropy.std())),
            'aleatoric': (float(aleatoric_entropy.mean()), float(aleatoric_entropy.std())),
            'epistemic': (float(epistemic_entropy.mean()), float(epistemic_entropy.std()))
        },
        'top-1': float(top_k_accuracy[0]),
        'top-2': float(top_k_accuracy[1]),
        'top-3': float(top_k_accuracy[2])
    }

    return results
