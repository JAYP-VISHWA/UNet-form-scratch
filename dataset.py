import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CarvanaDataset(Dataset):
    """Class for Caravan dataset

    :param Dataset: torch dataset Module
    :type Dataset: Torch module
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        """Constructor

        :param image_dir: Image directory
        :type image_dir: string
        :param mask_dir: Mask directory
        :type mask_dir: string
        :param transform: transforms for data, defaults to None
        :type transform: _type_, optional
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        """To define length of dataset

        :return: length of dataset
        :rtype: int
        """
        return len(self.images)

    def __getitem__(self, index):
        """To accesss dataset

        :param index: index of dataset element
        :type index: int
        :return: image and mask
        :rtype: tensors
        """
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(
            self.mask_dir, self.images[index].replace(".jpg", "_mask.gif")
        )
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
