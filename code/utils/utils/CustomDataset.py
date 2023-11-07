import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import json
import math
import argparse
import os
import random
import shutil
import time
import warnings
from PIL import Image, ImageDraw
import PIL


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_info, root_dir, transform=None):
        self.data_info = data_info
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_info["images"])

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.root_dir, self.data_info["images"][idx]["file_name"]
        )
        image = Image.open(img_name).convert("L")  # Ensure it's grayscale
        image_id = self.data_info["images"][idx]["id"]

        # Find all corresponding annotations for the image_id
        annotations = [
            ann for ann in self.data_info["annotations"] if ann["image_id"] == image_id
        ]

        # Initialize an empty mask
        mask = Image.new("L", image.size, 0)

        for ann in annotations:
            if not len(ann["segmentation"]) > 1:
                self.create_mask(mask, ann["segmentation"], image.size)
                # print(mask)
                # plt.imshow(mask, cmap='gray')  # 'cmap' is set to 'gray' for grayscale images
                # plt.axis('off')  # Optionally remove the axes
                # plt.show()

        # Convert PIL Images to tensors
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
            mask = to_tensor(mask)

        sample = {"image": image, "annotations": mask}

        return sample

    @staticmethod
    def create_mask(mask_img, segmentation, size):
        # Draw the segment on the mask image
        for segment in segmentation:
            ImageDraw.Draw(mask_img).polygon(segment, outline=1, fill=1)
