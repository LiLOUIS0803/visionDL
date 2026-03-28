import os
import csv
import argparse
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# Dataset
# ============================================================
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
    old_classes = dataset.classes[:]
    sorted_classes = sorted(old_classes, key=lambda x: int(x))
    new_class_to_idx = {cls: idx for idx, cls in enumerate(sorted_classes)}

    new_samples = []
    for path, old_idx in dataset.samples:
        class_name = old_classes[old_idx]
        new_idx = new_class_to_idx[class_name]
        new_samples.append((path, new_idx))

    dataset.classes = sorted_classes
    dataset.class_to_idx = new_class_to_idx
    dataset.samples = new_samples
    dataset.targets = [s[1] for s in new_samples]


def get_dataloaders(data_root, batch_size=32, num_workers=4, train=True):
    data_root = Path(data_root)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    image_size = 480

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.5, 1.0),
            ratio=(0.75, 1.3333),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), value="random"),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.05), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(data_root, "train"), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_root, "val"), transform=val_test_transform
    )
    test_dataset = TestDataset(
        os.path.join(data_root, "test"), transform=val_test_transform
    )

    fix_class_to_idx(train_dataset)
    fix_class_to_idx(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if not train:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return test_loader, train_dataset.classes

    return train_loader, val_loader, train_dataset.classes


# ============================================================
# CBAM
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ============================================================
# GeM Pooling
# ============================================================
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


# ============================================================
# Model: Multi-Scale CBAM + ResNeXt101
# ============================================================
class MultiScaleCBAMResNeXt(nn.Module):
    def __init__(self, num_classes=100, dropout=0.3):
        super().__init__()

        backbone = models.resnext101_32x8d(
            weights=models.ResNeXt101_32X8D_Weights.DEFAULT
        )

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)

        self.pool2 = GeM()
        self.pool3 = GeM()
        self.pool4 = GeM()

        feat_dim = 512 + 1024 + 2048

        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)

        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        f2 = self.pool2(self.cbam2(x2)).flatten(1)
        f3 = self.pool3(self.cbam3(x3)).flatten(1)
        f4 = self.pool4(self.cbam4(x4)).flatten(1)

        feat = torch.cat([f2, f3, f4], dim=1)
        return self.head(feat)


# ============================================================
# Model: Multi-Scale CBAM + ResNestRS101
# ============================================================

class MultiScaleCBAMResNetRS(nn.Module):
    def __init__(self, num_classes=100, dropout=0.3, model_name="resnetrs101.tf_in1k"):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=(2, 3, 4),
        )

        channels = self.backbone.feature_info.channels()

        self.cbam2 = CBAM(channels[0])
        self.cbam3 = CBAM(channels[1])
        self.cbam4 = CBAM(channels[2])

        self.pool2 = GeM()
        self.pool3 = GeM()
        self.pool4 = GeM()

        feat_dim = sum(channels)

        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        x2, x3, x4 = feats

        f2 = self.pool2(self.cbam2(x2)).flatten(1)
        f3 = self.pool3(self.cbam3(x3)).flatten(1)
        f4 = self.pool4(self.cbam4(x4)).flatten(1)

        feat = torch.cat([f2, f3, f4], dim=1)
        return self.head(feat)


# ============================================================
# CutMix / MixUp utilities
# ============================================================
def rand_bbox(size, lam):
    H, W = size[2], size[3]
    cut_rat = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    cy = torch.randint(0, H, (1,)).item()
    cx = torch.randint(0, W, (1,)).item()

    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    return y1, y2, x1, x2


def cutmix_data(images, targets, alpha=1.0):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = images.clone()
    y1, y2, x1, x2 = rand_bbox(images.size(), lam)
    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    lam = 1 - (y2 - y1) * (x2 - x1) / (images.size(-1) * images.size(-2))
    return mixed_images, targets, targets[index], lam


def mixup_data(images, targets, alpha=0.4):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[index]
    return mixed, targets, targets[index], lam


# ============================================================
# Logging / Curves
# ============================================================
def save_loss_curve(train_losses, val_losses, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_curve_path = save_dir / "loss_curve.png"
    plt.savefig(loss_curve_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Loss curve saved to {loss_curve_path}")


def save_accuracy_curve(train_accs, val_accs, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_accs) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    acc_curve_path = save_dir / "accuracy_curve.png"
    plt.savefig(acc_curve_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Accuracy curve saved to {acc_curve_path}")


def save_training_log(train_losses, val_losses, train_accs, val_accs, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log_path = save_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        for i in range(len(train_losses)):
            writer.writerow([
                i + 1,
                train_losses[i],
                val_losses[i],
                train_accs[i],
                val_accs[i],
            ])

    print(f"Training log saved to {log_path}")


# ============================================================
# Training
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)

        r = torch.rand(1).item()
        if r < 0.4:
            images_mixed, targets_a, targets_b, lam = cutmix_data(images, targets)
            outputs = model(images_mixed)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        elif r < 0.7:
            images_mixed, targets_a, targets_b, lam = mixup_data(images, targets)
            outputs = model(images_mixed)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100. * correct / total:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(loader, desc="Validating"):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, classes = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True,
    )
    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")

    model = MultiScaleCBAMResNetRS(
        num_classes=num_classes,
        dropout=args.dropout,
        model_name=args.model_name,
    ).to(device)
    # model = MultiScaleCBAMResNeXt(num_classes=num_classes, dropout=args.dropout).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    backbone_params = []
    new_params = []
    for name, param in model.named_parameters():
        if "cbam" in name or "head" in name or "pool" in name:
            new_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": new_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% | "
            f"LR: {lr_now:.6f}"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  -> New best model saved (epoch={best_epoch}, val acc={best_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  -> No improvement. EarlyStop counter: {patience_counter}/{args.early_stop_patience}")

        torch.save(model.state_dict(), save_dir / "last_model.pth")

        save_training_log(train_losses, val_losses, train_accs, val_accs, save_dir)
        save_loss_curve(train_losses, val_losses, save_dir)
        save_accuracy_curve(train_accs, val_accs, save_dir)

        if patience_counter >= args.early_stop_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}. Best epoch: {best_epoch}, best val acc: {best_acc:.2f}%")
            break

    print(f"\nTraining done. Best epoch: {best_epoch}, Best val acc: {best_acc:.2f}%")


# ============================================================
# Inference with TTA
# ============================================================
@torch.no_grad()
def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_loader, classes = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=False,
    )
    num_classes = len(classes)

    model = MultiScaleCBAMResNetRS(
        num_classes=num_classes,
        dropout=0.0,
        model_name=args.model_name,
    ).to(device)
    # model = MultiScaleCBAMResNeXt(num_classes=num_classes, dropout=0.0).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    image_size = 480

    tta_transforms = [
        transforms.Compose([
            transforms.Resize(int(image_size * 1.05), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]),
        transforms.Compose([
            transforms.Resize(int(image_size * 1.05), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]),
        transforms.Compose([
            transforms.Resize(int(image_size * 1.10), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]),
    ]

    test_root = os.path.join(args.data_root, "test")
    image_names = sorted(os.listdir(test_root))

    predictions = []

    for img_name in tqdm(image_names, desc="Inference (TTA)"):
        img_path = os.path.join(test_root, img_name)
        img = Image.open(img_path).convert("RGB")

        logits_sum = None
        for t in tta_transforms:
            x = t(img).unsqueeze(0).to(device)
            logits = model(x)
            logits_sum = logits if logits_sum is None else logits_sum + logits

        pred_idx = logits_sum.argmax(dim=1).item()
        pred_class = classes[pred_idx]
        predictions.append((os.path.splitext(img_name)[0], pred_class))

    output_path = Path(args.save_dir) / "prediction.csv"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_label"])
        writer.writerows(predictions)

    print(f"Predictions saved to {output_path}")


# ============================================================
# Evaluation utilities
# ============================================================
@torch.no_grad()
def evaluate_and_save_confusion_matrix(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_loader, classes = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True,
    )
    num_classes = len(classes)

    model = MultiScaleCBAMResNetRS(
        num_classes=num_classes,
        dropout=0.0,
        model_name=args.model_name,
    ).to(device)
    # model = MultiScaleCBAMResNeXt(num_classes=num_classes, dropout=0.0).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_targets = []
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(val_loader, desc="Evaluating"):
        images, targets = images.to(device), targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    val_loss = running_loss / total
    val_acc = 100.0 * correct / total

    print(f"Eval Loss: {val_loss:.4f}")
    print(f"Eval Acc : {val_acc:.2f}%")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))

    cm_csv_path = save_dir / "confusion_matrix.csv"
    with open(cm_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt/pred"] + classes)
        for i, row in enumerate(cm):
            writer.writerow([classes[i]] + row.tolist())

    fig_size = max(12, num_classes * 0.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=90,
        colorbar=False,
        values_format="d",
    )
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    cm_png_path = save_dir / "confusion_matrix.png"
    plt.savefig(cm_png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix csv saved to {cm_csv_path}")
    print(f"Confusion matrix image saved to {cm_png_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--early_stop_patience", type=int, default=20)

    parser.add_argument("--model_name", type=str, default="resnetrs101.tf_in1k")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "eval"])

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        inference(args)
    else:
        evaluate_and_save_confusion_matrix(args)
