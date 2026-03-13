import os
import copy
import time
from pathlib import Path
from xml.parsers.expat import model
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = sorted(os.listdir(root))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name

def fix_class_to_idx(dataset):
    old_classes = dataset.classes[:]   # 原本字串排序
    sorted_classes = sorted(old_classes, key=lambda x: int(x))  # 按數字排序

    old_class_to_idx = dataset.class_to_idx.copy()
    new_class_to_idx = {cls: idx for idx, cls in enumerate(sorted_classes)}

    dataset.classes = sorted_classes
    dataset.class_to_idx = new_class_to_idx

    new_samples = []
    for path, old_idx in dataset.samples:
        class_name = old_classes[old_idx]
        new_idx = new_class_to_idx[class_name]
        new_samples.append((path, new_idx))

    dataset.samples = new_samples
    dataset.targets = [s[1] for s in new_samples]
    

def get_dataloaders(data_root, batch_size=32, num_workers=4):
    data_root = Path(data_root)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(232),   
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_root, "val"), transform=val_test_transform)
    test_dataset = TestDataset(os.path.join(data_root, "test"), transform=val_test_transform)
    
    fix_class_to_idx(train_dataset)
    fix_class_to_idx(val_dataset)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader, test_loader, train_dataset.classes


def build_model(num_classes, device):
    # 載入 ImageNet V2 預訓練權重
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model.to(device)


def accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc

def inference(model, loader, device, class_names):
    model.eval()
    results = []

    with torch.no_grad():
        for images, names in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for name, pred in zip(names, preds):
                image_name = Path(name).stem
                pred_label = class_names[pred.item()]
                results.append((image_name, pred_label))

    df = pd.DataFrame(results, columns=["image_name", "pred_label"])
    df.to_csv("prediction.csv", index=False)
    print("Saved prediction.csv")

def train_model(
    data_root,
    save_path="best_resnet50.pth",
    epochs=20,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-4,
    num_workers=4,
    inference_only=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers
    )
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")

    model = build_model(num_classes=num_classes, device=device)

    if not inference_only:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        best_val_acc = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())

        for epoch in range(epochs):
            start_time = time.time()

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            scheduler.step()
            elapsed = time.time() - start_time

            print(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    "model_state_dict": best_model_wts,
                    "class_names": class_names,
                    "num_classes": num_classes,
                }, save_path)
                print(f"Saved best model to {save_path}")

        print(f"Best Val Acc: {best_val_acc:.4f}")

    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    class_names = checkpoint["class_names"]

    inference(model, test_loader, device, class_names)

    return model, class_names



if __name__ == "__main__":
    DATA_ROOT = "./data" 
    SAVE_PATH = "./best_resnet50_imagenetv2_100cls.pth"

    train_model(
        data_root=DATA_ROOT,
        save_path=SAVE_PATH,
        inference_only=True,
        epochs=20,
        batch_size=32,
        lr=1e-4,
        weight_decay=1e-4,
        num_workers=4
    )
