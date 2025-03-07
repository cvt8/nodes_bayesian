import torch.nn as nn
import torch


class ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.n_bins = n_bins

    def forward(self, probabilities, labels):
        confidences, predictions = torch.max(probabilities, dim=1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=probabilities.device)
        for bin_lower, bin_upper in zip(self.bin_boundaries[:-1], self.bin_boundaries[1:]):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
