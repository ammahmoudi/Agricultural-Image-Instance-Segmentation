import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
from PIL import Image
import numpy as np


def read_image_as_tensor(image_path):
    """
    Reads an image from a file and converts it to a tensor.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Tensor representation of the image.
    """
    img = Image.open(image_path).convert("RGB")
    return torch.tensor(np.array(img), dtype=torch.float32)


def read_mask_as_tensor(mask_path):
    """
    Reads a mask from a file and converts it to a tensor.

    Args:
        mask_path (str): Path to the mask file.

    Returns:
        torch.Tensor: Tensor representation of the mask.
    """
    mask = Image.open(mask_path)
    return torch.tensor(np.array(mask), dtype=torch.int32)


class PennFudanDataset(Dataset):
    """
    Custom dataset for Penn-Fudan pedestrian segmentation.

    This class handles loading images and their corresponding instance segmentation masks,
    and provides them in a format suitable for training or evaluation.

    Args:
        root (str): Root directory of the dataset containing `images` and `leaf_instances` directories.
        transforms (callable): Transformations to apply to images and masks.
    """
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "leaf_instances"))))

    def __getitem__(self, idx):
        """
        Fetches an image and its corresponding target (masks, bounding boxes, etc.) by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple (image, target) where target is a dictionary containing:
                - boxes: Bounding boxes for objects in XYXY format.
                - masks: Binary masks for each object.
                - labels: Class labels (all set to 1 as only one class exists).
                - image_id: The index of the image.
                - area: The area of each bounding box.
                - iscrowd: Crowd annotations (set to 0 for all objects).
        """
        # Load image and mask paths
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "leaf_instances", self.masks[idx])

        # Read image and mask
        img = read_image(img_path)
        mask = read_mask_as_tensor(mask_path)

        # Get unique object IDs and exclude the background (ID = 0)
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[obj_ids > 0]
        num_objs = len(obj_ids)

        # Generate binary masks for each object
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # Compute bounding boxes
        boxes = masks_to_boxes(masks)

        # Ensure bounding boxes have valid dimensions
        for box in boxes:
            if box[0] <= 0:
                box[0] = 1
            if box[1] <= 0:
                box[1] = 1
            if box[2] <= box[0]:
                box[2] = box[0] + 1
            if box[3] <= box[1]:
                box[3] = box[1] + 1

        # Create additional annotations
        labels = torch.ones((num_objs,), dtype=torch.int64)  # Class labels (1 for all objects)
        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # Area of each bounding box
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)  # Crowd annotation (0 for all objects)

        # Convert image and annotations to torchvision tensor objects
        img = tv_tensors.Image(img)
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        # Apply transformations if provided
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.imgs)

class PennFudanTestDataset(Dataset):
    """
    Dataset for testing without masks or other annotations.
    Loads only images from the specified directory.
    """
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = read_image(img_path)
        if self.transforms:
            img = self.transforms(img)
        return img, {}

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    """
    Get transformations to apply on the dataset.

    Args:
        train (bool): Whether to apply training-specific transformations.

    Returns:
        torchvision.transforms.Compose: Composed transformations.
    """
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def collate_fn(batch):
    """
    Collate function for DataLoader to handle batches of data.

    Args:
        batch (list): A list of samples from the dataset.

    Returns:
        tuple: Collated batch in the form of tuples.
    """
    return tuple(zip(*batch))
