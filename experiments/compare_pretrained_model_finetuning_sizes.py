import time
import gc
import os
import sys
import torch
import subprocess
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader


# Custom imports assuming they are in the parent directory
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import test_visualization
from data_augmentation import DataAugmentation
from metrics import pixel_wise_accuracy, evaluate_model_performance
from losses import contrastive_loss, dice_loss
from models import Decoder, MaskedAutoEncoder
from data_utils import ContDataset, Transform


# Configuration for datasets
batch_size = 64
dataset_config = {
    'image_size': (224, 224),
    'data_path': '../datasets/data',
    'aug_data_path': '../datasets/aug_data'
}

pre_train = {
    'num_samples': 8000,  # Size of the pre-trained dataset
    'epochs': 10,  # Total epochs for pre-training
    'learning_rate': 1e-3  # Learning rate in the pre-training phase
}

# Split configurations for fine-tuning
fine_tune_dataset_split = {
    'train_ratio': 0.8,  # Fine-tuning the scale of the training dataset
    'test_ratio': 0.2,  # Fine-tuning the scale of the test dataset
}

# Training configuration for fine-tuning
fine_tune_training_config = {
    'batch_size': 64,
    'shuffle_train': True,
    'shuffle_test': False,
    'training_epochs': 10,  # Fine-tuning phase training epochs
    'learning_rate': 1e-4  # Learning rate in the pre-training phase
}

# Fine-tuning dataset size ratios
# Example sizes: 10%, 50%, and 100% of the dataset
dataset_sizes = [0.1, 0.5, 1.0]


# # Pre-Train


# Unzip dataset if not already present
if not os.path.exists(dataset_config['data_path']):
    subprocess.run(f'unzip ../datasets/data.zip -d {"../datasets"}',
                   shell=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
else:
    print("The data folder already exists, no need to unzip it again")

# Initialize data augmentation module
augmentor = DataAugmentation(dataset_config['data_path'],
                             dataset_config['aug_data_path'],
                             pre_train['num_samples'])
augmentor.augment_images()

# List files and count them in each directory using subprocess
data_files_count = subprocess.check_output(
    f'ls -1 {dataset_config["data_path"]} | wc -l', shell=True).strip().decode()
aug_data_files_count = subprocess.check_output(
    f'ls -1 {dataset_config["aug_data_path"]} | wc -l', shell=True).strip().decode()
print(f"Number of files in data directory: {data_files_count}")
print(
    f"Number of files in aug_data directory - Pre-training dataset: {aug_data_files_count}")

# Define a transform to convert the images to PyTorch tensors and any other desired transformations
transform = transforms.Compose([
    # Resize the image to 224x224 pixels.
    transforms.Resize(dataset_config['image_size']),
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
])

# Load dataset and create dataloader
dataset = ContDataset(folder_path=dataset_config['aug_data_path'],
                      folder_path1=dataset_config['data_path'],
                      transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Setup model and training devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = vit_b_16(pretrained=False).to(device)
decoder = Decoder(1000, 512, 3 * 224 * 224).to(device)
pre_model = MaskedAutoEncoder(encoder, decoder).to(device)
optimizer = optim.Adam(pre_model.parameters(), lr=pre_train['learning_rate'])
mask = torch.rand(size=(1, 3, 224, 224)) > 0.5
mask = mask.to(device)
scaler = torch.cuda.amp.GradScaler()


# Start the pre-training phase
print("Starting the pre-training phase...")

for epoch in range(pre_train['epochs']):
    start_time = time.time()
    pre_model.train()
    epoch_losses = []  # Collect losses for each batch to calculate epoch average

    for x, z1, z2 in dataloader:
        inputs, x1, x2 = x.to("cuda"), z1.to("cuda"), z2.to("cuda")
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            try:
                p1, p2 = pre_model(x1, mask).reshape(64, 3, 224, 224), pre_model(
                    x2, mask).reshape(64, 3, 224, 224)
                loss = contrastive_loss(inputs, p1, p2)
                epoch_losses.append(loss.item())

            except Exception as e:
                continue  # Skip the backward pass and optimizer step if an error occurred

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # Calculate and print the average loss for the epoch
    epoch_avg_loss = sum(epoch_losses) / \
        len(epoch_losses) if epoch_losses else float('inf')
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(
        f'Epoch {epoch+1}, Avg. Loss: {epoch_avg_loss:.3f}, Duration: {epoch_duration:.2f} seconds')

print("Pre-training phase completed.")


# # Fine-tuning of pre-trained models （10%, 50% and 100% fine-tuned data sets）


transform = Transform(image_size=dataset_config['image_size'])
full_dataset = torchvision.datasets.OxfordIIITPet(root='../datasets',
                                                  target_types='segmentation',
                                                  transforms=transform,
                                                  download=True)

# Function to perform fine-tuning


def fine_tune_model(data_ratio):

    # Define the size of training and testing datasets
    total_size = len(full_dataset)
    used_data_size = int(total_size * data_ratio)
    train_size = int(used_data_size * fine_tune_dataset_split['train_ratio'])
    test_size = used_data_size - train_size

    # Split the dataset
    indices = torch.randperm(total_size).tolist()
    used_indices = indices[:used_data_size]
    train_indices = used_indices[:train_size]
    test_indices = used_indices[train_size:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    print(f"\nFine-tuning Original dataset size: {total_size}")
    print(
        f"Fine-tuning Used dataset size: {used_data_size} ({data_ratio*100}%)")
    print(f"Fine-tuning Training dataset size: {len(train_dataset)}")
    print(f"Fine-tuning Testing dataset size: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=fine_tune_training_config['batch_size'], shuffle=fine_tune_training_config['shuffle_train'], drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=4,
                             shuffle=fine_tune_training_config['shuffle_test'], drop_last=True)

    mask = torch.ones(size=(1, 3, 224, 224)).to(device)
    fine_model_with_pre = pre_model.to(device)
    optimizer = optim.Adam(fine_model_with_pre.parameters(
    ), lr=fine_tune_training_config['learning_rate'])

    # Fine-tuning the model
    for epoch in range(fine_tune_training_config['training_epochs']):
        start_time = time.time()
        fine_model_with_pre.train()
        for x, y in train_loader:
            inputs, targets = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = fine_model_with_pre(inputs, mask)
            if preds.size(0) == inputs.size(0):
                batch_size = preds.shape[0]
                preds = preds.reshape(batch_size, 3, 224, 224)
                loss = dice_loss(preds, targets)
                accuracy = pixel_wise_accuracy(preds, targets)
                loss.backward()
                optimizer.step()

        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Data ratio {data_ratio * 100}%: Epoch {epoch+1}, Loss: {loss.item():.3f}, Accuracy: {accuracy:.3f}, Duration: {epoch_duration:.2f} seconds")

    # Evaluate the performance
    evaluate_model_performance(fine_model_with_pre, test_loader, device,
                               mask, f"fine-tuned model with data ratio {data_ratio * 100}%")

    test_visualization(fine_model_with_pre, test_loader, mask, device,
                       f"fine-tuned model with data ratio {data_ratio * 100}%", "../images/compare_pretrained_model_finetuning_sizes")

    fine_model_with_pre.to('cpu')
    torch.cuda.empty_cache()
    gc.collect()


# Iterate over different dataset sizes
for size in dataset_sizes:
    fine_tune_model(size)
