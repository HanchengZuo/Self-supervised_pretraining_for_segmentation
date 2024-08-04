import os
import random
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from typing import Callable, Tuple


class ContDataset(Dataset):
    """
    Prepare dataset for contrastive learning where each augmented image is paired with its original image.
    The names of the augmented images end with `'_left.jpg'` or `'_right.jpg'`, and find the corresponding
    original image by removing these suffixes.
    """

    def __init__(self, folder_path: str, folder_path1: str, transform: Callable | None = None):
        """
        Initialize the class.
        
        Args:
            folder_path (string): Path to the folder with augmented images.
            folder_path1 (string): Path to the folder with original images.
            transform (callable, optional): Optional transform to be applied on a sample. Default to `None`.
        """
        self.folder_path = folder_path
        self.folder_path1 = folder_path1
        self.transform = transform
        self.augmented_filenames = sorted(os.listdir(folder_path))
        self.original_filenames = sorted(os.listdir(folder_path1))

        # Create a mapping from augmented to original filenames
        self.mapping = {}
        for filename in self.augmented_filenames:
            base_name = filename.replace(
                '_left.jpg', '').replace('_right.jpg', '')
            self.mapping[filename] = base_name + '.jpg'

    def __len__(self):
        return len(self.augmented_filenames)

    def __getitem__(self, idx):
        # Get the filename of the augmented image
        augmented_filename = self.augmented_filenames[idx]
        augmented_img_path = os.path.join(self.folder_path, augmented_filename)
        augmented_image = Image.open(augmented_img_path)

        # Find the corresponding original image
        original_filename = self.mapping[augmented_filename]
        original_img_path = os.path.join(self.folder_path1, original_filename)
        original_image = Image.open(original_img_path)

        # Select a random negative sample that is not the same as the original
        neg_idx = random.choice([i for i in range(
            len(self.original_filenames)) if self.original_filenames[i] != original_filename])
        neg_img_path = os.path.join(
            self.folder_path1, self.original_filenames[neg_idx])
        negative_image = Image.open(neg_img_path)

        # Apply transformations if any
        if self.transform:
            augmented_image = self.transform(augmented_image)
            original_image = self.transform(original_image)
            negative_image = self.transform(negative_image)

        return augmented_image, original_image, negative_image


class MockContDataset(Dataset):
    """Prepare random dataset for contrastive learning."""

    def __init__(self, num_samples: int, image_size: Tuple[int, int], transform: bool | None = None):
        """
        Initialize the class.

        Args:
            num_samples (int): Number of samples in the dataset.
            image_size (tuple[int, int]): Size of the generated images (height, width).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        # Return the total number of image pairs
        # There is one less pair than the number of images
        return self.num_samples - 1

    def __getitem__(self):
        # Generate random images
        image_0 = Image.fromarray(np.random.randint(
            0, 255, (self.image_size[0], self.image_size[1], 3), dtype=np.uint8))
        image_1 = Image.fromarray(np.random.randint(
            0, 255, (self.image_size[0], self.image_size[1], 3), dtype=np.uint8))
        image_2 = Image.fromarray(np.random.randint(
            0, 255, (self.image_size[0], self.image_size[1], 3), dtype=np.uint8))

        if self.transform:
            image_0 = self.transform(image_0)
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        # No label is provided, just the pair of images
        return image_0, image_1, image_2


class Transform:
    """Transform the images and correspounding pixel-wise lables to a pre-defined shape"""

    def __init__(self, image_size=(224, 224)):
        """
        Initialize the class.

        Args:
            image_size (tuple[int]): (width, height) of the output shape
        """
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def __call__(self, img, mask):
        img = self.transform(img)
        # Assuming mask is a PIL image, resize it as the img and convert to tensor.
        mask = F.resize(mask, self.image_size, Image.NEAREST)
        # Masks don't need normalization but need to be torch tensors
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        return img, mask


def check_image_shapes(folder_path: str):
    """
    Check the modes of data.

    Args:
        folder_path (str): path of the folder.

    Returns:
        modes (dict): dict with modes as keys and number of files as values.
        filenames (list): filenmames with the images whose modes are not RGB.
    """

    modes = {}  # record number of images under each mode
    filenames = []  # record images names that are not RGB
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                # Get image channels (mode)
                mode = img.mode

                if mode not in modes:
                    modes[mode] = 1
                else:
                    modes[mode] += 1

                if mode != 'RGB':
                    filenames.append(filename)

    return modes, filenames
