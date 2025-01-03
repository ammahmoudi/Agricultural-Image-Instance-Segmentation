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
    """Reads an image from a file and converts it to a tensor."""
    img = Image.open(image_path).convert("RGB")
    return torch.tensor(np.array(img), dtype=torch.float32)


def read_mask_as_tensor(mask_path):
    """Reads a mask from a file and converts it to a tensor."""
    mask = Image.open(mask_path)
    return torch.tensor(np.array(mask), dtype=torch.int32)


class PennFudanDataset(Dataset):
    """Dataset for PennFudan data, containing images and instance masks."""

    def __init__(self, root, transforms):
        """
        Initialize the dataset.
        :param root: Root directory containing images and masks.
        :param transforms: Transformations to apply on images and masks.
        """
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "leaf_instances"))))

    def __getitem__(self, idx):
        """
        Get an image and its corresponding mask.
        :param idx: Index of the image and mask to get.
        :return: Tuple (image, target) where target is a dictionary of boxes, masks, labels, image_id, area, and iscrowd.
        """
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "leaf_instances", self.masks[idx])
        img = read_image(img_path)
        mask = read_mask_as_tensor(mask_path)
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[obj_ids > 0]
        num_objs = len(obj_ids)
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
        boxes = masks_to_boxes(masks)

        # Ensure box coordinates are valid
        boxes = torch.clamp(boxes, min=1)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        img = tv_tensors.Image(img)
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes, format="XYXY", canvas_size=F.get_size(img)
            ),
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.imgs)


def get_transform(train):
    """Get the transformations to apply on the dataset."""
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def collate_fn(batch):
    """Collate function for DataLoader to handle batches."""
    return tuple(zip(*batch))
