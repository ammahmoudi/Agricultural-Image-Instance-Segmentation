# main.py
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from engine import train_one_epoch, evaluate
from dataset import PennFudanDataset, get_transform, collate_fn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Set up dataset and data loader
dataset = PennFudanDataset('PhenoBench/train/', get_transform(train=True))
dataset_test = PennFudanDataset('PhenoBench/val/', get_transform(train=False))

# Use a subset of the dataset for faster iterations
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:10])
dataset_test = torch.utils.data.Subset(dataset_test, indices[:5])

data_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

data_loader_test = DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)

# Load the model and transfer it to the device
model = maskrcnn_resnet50_fpn(weights="DEFAULT")
model = model.to(device)

# Set up optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

num_epochs = 1

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device=device)

print("That's it!")

# Save the trained model
torch.save(model.state_dict(), 'model_2.pth')

# Load the model for inference
# model.load_state_dict(torch.load('model.pth'))
# model.eval()
# model = model.to(device)
