import os
import random
import cv2
import glob
import shutil
import imgaug.augmenters as iaa
import numpy as np

class DataAugmentation:
    """Class for data augmentation"""

    def __init__(self, original_path: str, augmented_path: str, num_images: int):
        """Initialize the class.

        Args:
            original_path (str): The path to the directory containing the original images.
            augmented_path (str): The path to the directory where augmented images will be saved.
            num_images (int): Number of images to be processed for augmentation.
        """
        self.original_path = original_path
        self.augmented_path = augmented_path
        self.num_images = num_images
        # Create augmented path directory if it doesn't exist
        os.makedirs(self.augmented_path, exist_ok=True)
        self._clear_directory(self.augmented_path)
        # List of augmentation functions from the imgaug library
        self.funs = [iaa.Fliplr(1), iaa.Flipud(1), iaa.Rotate(90),
                     iaa.GaussianBlur(), iaa.ChannelShuffle(0.35),
                     iaa.ChangeColorTemperature(
                         (1100, 10000)), iaa.Add((-40, 40)),
                     iaa.Dropout(p=(0, 0.2)), iaa.pillike.Autocontrast(),
                     iaa.PadToFixedSize(width=100, height=100)]

    def _clear_directory(self, path: str):
        """
        Clears all files in the specified directory.
        
        Args:
            path (str): path where all files will be cleared.
        """

        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    def augment_images(self):
        """Perform data augmentation on the specified number of images."""

        # Fetch image paths and select a specified number for processing
        img_paths = glob.glob(os.path.join(self.original_path, '*.*'))
        selected_paths = img_paths[:self.num_images]  # 取前num_images张图片
        print("Data Augmentation in progress...")

        for img_path in selected_paths:
            # Read the image
            img = cv2.imread(img_path, 1)
            # Randomly select an augmentation function
            func = random.choice(self.funs)
            # Apply the augmentation
            img_r = func(image=img)

            # Get the height and width of the augmented image
            h, w = img_r.shape[:2]
            # Extract the original filename
            original_filename = os.path.splitext(os.path.basename(img_path))[0]

            # Perform left augmentation and save the image
            left_img = self._augment_single_image(img_r, h, w)
            left_save_p = os.path.join(
                self.augmented_path, f'{original_filename}_left.jpg')
            cv2.imwrite(left_save_p, left_img)

            # Perform right augmentation and save the image
            right_img = self._augment_single_image(img_r, h, w, is_right=True)
            right_save_p = os.path.join(
                self.augmented_path, f'{original_filename}_right.jpg')
            cv2.imwrite(right_save_p, right_img)

        print("Data Augmentation Done!")

    def _augment_single_image(self, img: np.ndarray, h: int, w: int, is_right: bool | None = False):
        """
        Apply augmentation to a single image by adding a mask block.

        Args:
            img (array): The image to be augmented.
            h (int): Height of the image.
            w (int): Width of the image.
            is_right (bool): Flag to determine whether to augment the right side of the image. Default to `False`.

        Returns:
            array: The augmented image.
        """

        # Make a copy of the image to avoid modifying the original
        augmented_img = img.copy()
        # Get random block parameters
        bh, bw, size1, size2 = self._random_block_params(h, w)
        # Add a block to the right or left part of the image
        if is_right:
            augmented_img[bh:bh + size1, w // 2 +
                          bw:w // 2 + bw + size2, :] = 114
        else:
            augmented_img[bh:bh + size1, bw:bw + size2, :] = 114
        return augmented_img

    def _random_block_params(self, h: int, w: int):
        """
        Generate random parameters for the block to be added to the image.

        Args:
            h (int): Height of the image.
            w (int): Width of the image.

        Returns:
            bh, bw, size1, size2: args for the block (bh, bw, size1, size2)
        """

        bh = random.randint(5, h // 3)
        bw = random.randint(5, w // 3)
        size1 = random.randint(100, 130)
        size2 = random.randint(100, 130)
        return bh, bw, size1, size2
