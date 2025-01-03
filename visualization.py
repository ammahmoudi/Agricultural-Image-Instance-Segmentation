# visualization.py
import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks
from dataset import read_image
from engine import get_transform

def visualize_predictions(model, image_path, colors, device):
    model.eval()
    image = read_image(image_path)
    eval_transform = get_transform(train=False)
    with torch.no_grad():
        x = eval_transform(image)
        predictions = model([x.to(device), ])
        pred = predictions[0]

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    masks = (pred["masks"] > 0.8).squeeze(1)
    output_image = draw_segmentation_masks(image, masks, alpha=0.5, colors=colors)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig("output_image.png")
    plt.show()

    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig('pred.png')
    # plt.savefig('pred.jpg')
    # plt.savefig('pred.svg')
