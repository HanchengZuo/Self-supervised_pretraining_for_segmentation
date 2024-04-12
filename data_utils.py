import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F


class ContDataset(Dataset):
    """
    Prepare dataset for contrastive learning

    Args:
      folder_path (string): Path to the folder with images.
      transform (callable, optional): Optional transform to be applied
          on a sample.
    """

    def __init__(self, folder_path, folder_path1, transform=None):
        self.folder_path = folder_path
        self.folder_path1 = folder_path1
        self.image_filenames = [f for f in sorted(os.listdir(
            folder_path)) if os.path.isfile(os.path.join(folder_path, f))]
        self.image_filenames1 = [f for f in sorted(os.listdir(
            folder_path1)) if os.path.isfile(os.path.join(folder_path1, f))]
        self.transform = transform

    def __len__(self):
        # Return the total number of image pairs
        # There is one less pair than the number of images
        return len(self.image_filenames) - 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_0 = os.path.join(
            self.folder_path1, self.image_filenames1[idx])
        img_name_1 = os.path.join(self.folder_path, self.image_filenames[idx])
        img_name_2 = os.path.join(
            self.folder_path, self.image_filenames[idx + 1])

        image_0 = Image.open(img_name_0)
        image_1 = Image.open(img_name_1)
        image_2 = Image.open(img_name_2)

        if self.transform:
            image_0 = self.transform(image_0)
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        # No label is provided, just the pair of images
        return image_0, image_1, image_2


class MockContDataset(Dataset):
    """prepare random dataset for contrastive learning

    Args:
      num_samples (int): Number of samples in the dataset.
      image_size (tuple): Size of the generated images (height, width).
      transform (callable, optional): Optional transform to be applied
          on a sample.
    """

    def __init__(self, num_samples, image_size, transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        # Return the total number of image pairs
        # There is one less pair than the number of images
        return self.num_samples - 1

    def __getitem__(self, idx):
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
    """
    Transform the images and correspounding pixel-wise lables to a pre-defined shape

    Args:
        image_size (tuple[int]): (width, height) of the output shape
    """

    def __init__(self, image_size=(224, 224)):
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


def check_image_shapes(folder_path):
    """
    check the modes of data

    Args:
        folder_path: path of the folder

    Returns:
        modes: dict with modes as keys and number of files as values
        filennames: filenmames with the images whose modes are not RGB
    """

    modes = {}  # record number of images under each mode
    filenames = []  # record images names that are not RGB
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                # Get image dimensions and channels (mode)
                width, height = img.size
                mode = img.mode

                if mode not in modes:
                    modes[mode] = 1
                else:
                    modes[mode] += 1

                if mode != 'RGB':
                    filenames.append(filename)

    return modes, filenames
