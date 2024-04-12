import torch
import torch.nn as nn
import torch.nn.functional as Func


class ContrastiveLoss(nn.Module):
    """
    Contrastive learing loss

    Args:
        margin: margin for contrastive loss
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Calculate the positive pair distance
        pos_dist = Func.pairwise_distance(anchor, positive)
        # Calculate the negative pair distance
        neg_dist = Func.pairwise_distance(anchor, negative)

        # Calculate the contrastive loss
        loss = torch.mean((pos_dist ** 2) +
                          torch.clamp(self.margin - neg_dist, min=0.0) ** 2)

        return loss


contrastive_loss = ContrastiveLoss(margin=1.0)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth=1.0):
    '''
    Calculate dice loss

    Args:
        pred: output of model, (batch_size, num_class, width, height)
        target: true prediction for eac pixel, (batch_size, width, height)
        smooth: smooth parameter for dice loss

    Return:
        dice_loss: dice loss
    '''

    _, num_classes, _, _ = pred.shape
    target = target - 1  # convert values from 1-3 to 0-2

    assert target.min() >= 0 and target.max(
    ) < num_classes, 'target contains invalid class indices'

    # Convert logits to probabilities
    pred = torch.nn.functional.softmax(pred, dim=1)

    # convert targets, trimap, to one shot
    # (batch_size, width, height) -> (batch_size, 3, width, height)
    targets_one_hot = torch.nn.functional.one_hot(
        target, num_classes).permute(0, 3, 1, 2)
    targets_one_hot = targets_one_hot.type_as(pred)

    # calculate dice_score for each (batch,class)
    # sum over width and height
    intersection = torch.sum(pred * targets_one_hot, dim=(2, 3))
    union = torch.sum(pred, dim=(2, 3)) + \
        torch.sum(targets_one_hot, dim=(2, 3))
    dice_score = (2. * intersection + smooth) / (union + smooth)

    dice_loss = 1. - dice_score.mean()  # aveerage over batch and num_classes

    return dice_loss
