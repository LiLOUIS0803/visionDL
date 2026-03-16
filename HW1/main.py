import os
import copy
import time
import argparse
from pathlib import Path
from xml.parsers.expat import model
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights, ResNet152_Weights, resnet152
from sklearn.metrics import confusion_matrix

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
    

def get_dataloaders(data_root, batch_size=32, num_workers=4, train=True):
    data_root = Path(data_root)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    image_size = 320

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.6, 1.0),
            ratio=(0.75, 1.3333)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(360),
        transforms.CenterCrop(image_size),
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
    if not train:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader, train_dataset.classes
    return train_loader, val_loader, train_dataset.classes


def build_model(num_classes, device):
    # 載入 ImageNet V2 預訓練權重
    
    # ResNet-50 0.92
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    
    # ResNet-152
    # weights = ResNet152_Weights.IMAGENET1K_V2
    # model = resnet152(weights=weights)

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

    for images, labels in tqdm(loader, desc="Training"):
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
def evaluate(model, loader, criterion, device, class_names=None, save_cm_path=None, show_cm=False):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    
    if show_cm:
        cm = confusion_matrix(all_labels, all_preds)

        print("\nConfusion Matrix:")
        print(cm)

        if class_names is not None:
            df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
            print("\nConfusion Matrix DataFrame:")
            print(df_cm)

        if save_cm_path is not None:
            fig_size = max(8, len(cm) * 0.8)
            plt.figure(figsize=(fig_size, fig_size))
            plt.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.colorbar()

            tick_marks = range(len(cm))
            if class_names is not None:
                plt.xticks(tick_marks, class_names, rotation=90)
                plt.yticks(tick_marks, class_names)
            else:
                plt.xticks(tick_marks, tick_marks, rotation=90)
                plt.yticks(tick_marks, tick_marks)

            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

            # 在格子中間標數字
            thresh = cm.max() / 2.0 if cm.max() > 0 else 0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(
                        j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black"
                    )

            plt.tight_layout()
            plt.savefig(save_cm_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved confusion matrix to {save_cm_path}")

    return epoch_loss, epoch_acc

def inference(data_root, checkpoint='best.pth', device=None):
    data_root = Path(data_root)
    test_loader, class_names = get_dataloaders(data_root=data_root, train=False)
    
    model = build_model(num_classes=len(class_names), device=device)
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    class_names = checkpoint["class_names"]
    
    model.eval()
    results = []

    with torch.no_grad():
        for images, names in test_loader:
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
    label_smoothing=0.1,
    freeze_epochs=5,
    device=None,
): 
    train_loader, val_loader, class_names = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers
    )
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    
    model = build_model(num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True
    
    for epoch in range(epochs):
        start_time = time.time()

        if epoch == freeze_epochs:  
            for param in model.parameters():
                param.requires_grad = True
        
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

    return model, class_names

def evaluate_checkpoint(data_root, checkpoint_path, batch_size, num_workers, device):
    _, val_loader, class_names = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers
    )

    model = build_model(num_classes=len(class_names), device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluate(
        model=model,
        loader=val_loader,
        criterion=nn.CrossEntropyLoss(label_smoothing=0.1),
        device=device,
        class_names=class_names,
        save_cm_path="confusion_matrix.png",
        show_cm=True,
    )

if __name__ == "__main__":
    DATA_ROOT = "./data" 
    SAVE_PATH = "./best.pth"
    
    argparser = argparse.ArgumentParser(description="Train and Inference for Image Classification") 
    argparser.add_argument("--data_root", type=str, default=DATA_ROOT, help="Path to dataset root")
    argparser.add_argument("--save_path", type=str, default=SAVE_PATH, help="Path to save the best model checkpoint")
    argparser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    argparser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation")
    argparser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    argparser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    argparser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    argparser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    argparser.add_argument("--train", action="store_true", help="Whether to train the model")
    argparser.add_argument("--inference", action="store_true", help="Whether to run inference after training")
    argparser.add_argument("--freeze_epochs", type=int, default=5, help="Number of epochs to freeze backbone during training")
    argparser.add_argument("--eval_cm", action="store_true", help="Evaluate checkpoint and save confusion matrix")
    args = argparser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.train:
        train_model(
            data_root=args.data_root,
            save_path=args.save_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            num_workers=args.num_workers, 
            freeze_epochs=args.freeze_epochs,
            device=device,
        )

    if args.inference:
        inference(data_root=args.data_root, checkpoint=args.save_path, device=device)
        
    if args.eval_cm:
        evaluate_checkpoint(
            data_root=args.data_root,
            checkpoint_path=args.save_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device
        )