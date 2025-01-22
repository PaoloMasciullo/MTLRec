import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    def __init__(self, np_ratio=None):
        super(BPRLoss, self).__init__()
        self.np_ratio = np_ratio  # Number of negative samples per positive sample

    def forward(self, y_pred, y_true, reduction='mean'):
        """
        y_true: Tensor of labels, with value 1 for positive and 0 for negative.
        y_pred: Tensor of predicted scores.
        """
        # Separate positive and negative scores.
        pos_scores = y_pred[y_true == 1]
        neg_scores = y_pred[y_true == 0]

        if self.np_ratio is not None:
            # Randomly sample negative scores for each positive score.
            num_neg_samples = int(self.np_ratio * pos_scores.size(0))
            neg_scores = neg_scores[torch.randint(0, neg_scores.size(0), (num_neg_samples,))]

        # Compute pairwise differences between positive and negative scores.
        diff = pos_scores.view(-1, 1) - neg_scores.view(1, -1)

        # Apply the sigmoid function for the log-sigmoid loss.
        loss = -torch.log(torch.sigmoid(diff))

        # Reduce the loss based on the reduction method.
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Focal Loss is a modification of the standard cross-entropy loss, designed to address class imbalance
        by reducing the relative loss for well-classified examples (those with high confidence predictions).
        This allows the model to focus on learning from hard-to-classify examples.
        :param alpha: Balancing factor for class weights (default: 1.0).
        :param gamma: Focusing parameter (default: 2.0).
        :param reduction: Reduction type ('mean', 'sum', or 'none').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: Model outputs (sigmoid for binary, log-softmax for multiclass).
        :param targets: Ground truth labels.
        """
        if inputs.shape[-1] > 1:  # Multiclass classification
            # Convert log-softmax outputs to probabilities
            probs = torch.exp(inputs)
            targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1)).to(inputs.dtype)
        else:  # Binary classification
            # Sigmoid outputs are already probabilities
            probs = inputs
            targets_one_hot = targets.unsqueeze(-1)

        # Compute p_t
        pt = probs * targets_one_hot + (1 - probs) * (1 - targets_one_hot)

        # Compute focal weight
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # Compute focal loss
        loss = focal_weight * -torch.log(pt.clamp(min=1e-8))  # Avoid log(0)

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
