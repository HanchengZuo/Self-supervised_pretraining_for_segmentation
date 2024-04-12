import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def pixel_wise_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    '''calculate pixe-wise accuracy

    Args:
        pred: output (logits) of model, (batch_size, num_classes, width,height)
        target: ground truth, trimap
    Returns:
        accuracy: pixel accuracy
    '''

    pred_tri = torch.argmax(pred, axis=1)
    target = target - 1

    accuracy = torch.sum(pred_tri == target) / pred_tri.numel()

    return accuracy.item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6):
    """
    Calculate the Intersection over Union (IoU) score.

    Args:
        pred (torch.Tensor): The output of the model. Shape: (batch_size, num_classes, width, height).
        target (torch.Tensor): The ground truth labels for each pixel. Shape: (batch_size, width, height).
        smooth (float): A small value to prevent division by zero.

    Returns:
        float: The average IoU score across all classes and all batches.
    """
    _, num_classes, _, _ = pred.shape
    target = target - 1  # Convert values from 1-3 to 0-2, assuming target values start from 1

    assert target.min() >= 0 and target.max(
    ) < num_classes, 'Target contains invalid class indices'

    # Convert logits to probabilities
    pred = torch.nn.functional.softmax(pred, dim=1)

    # Convert targets to one-hot encoding
    # Shape change: (batch_size, width, height) -> (batch_size, num_classes, width, height)
    target_one_hot = torch.nn.functional.one_hot(
        target, num_classes).permute(0, 3, 1, 2)
    target_one_hot = target_one_hot.type_as(pred)

    # Calculate intersection and union for each batch and class
    # Sum over width and height
    intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
    union = torch.sum(pred + target_one_hot, dim=(2, 3)) - \
        intersection  # Ensure to subtract intersection once

    # Calculate IoU and avoid division by zero
    iou = (intersection + smooth) / (union + smooth)

    # Average over all classes and batches
    return iou.mean().item()


def evaluate_model_performance(model:nn.Module, 
                               dataloader:DataLoader, 
                               device:torch.device, 
                               mask:torch.Tensor, 
                               model_description:str):
    """
    Evaluate the model on given dataloader to compute accuracy and IoU score.

    Args:
        model (nn.Moudle): The PyTorch model to evaluate.
        dataloader (DateLoader): The DataLoader containing the test dataset.
        device (torch.device): The device on which the computations are performed.
        mask (Tensor): The mask tensor applied to inputs if necessary.
        model_description (str): Description of the model phase for output clarity.

    Returns:
        None: Prints the accuracy and IoU directly.
    """

    model.eval()  # Set the model to evaluation mode
    acc = 0
    iou_total = 0
    for x, y in dataloader:
        inputs, targets = x.to(device), y.to(device)
        preds = model(inputs, mask)
        batch_size = preds.shape[0]
        preds = preds.reshape(batch_size, 3, 224, 224)

        # Calculate accuracy
        acc += pixel_wise_accuracy(preds, targets)

        # Calculate IoU
        iou = iou_score(preds, targets)
        iou_total += iou

    # Calculate the average scores over all batches
    accuracy = acc / len(dataloader)
    average_iou = iou_total / len(dataloader)

    # Print the results
    print(
        f'Accuracy of {model_description} on the fine-tuning test dataset: {accuracy:.2f}')
    print(
        f'IoU score of {model_description} on the fine-tuning test dataset: {average_iou:.2f}')
