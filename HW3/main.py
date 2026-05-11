import argparse
import json
import os
import random
from collections import OrderedDict
from typing import Dict, List, Optional

import cv2
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import (
    MaskRCNN,
    MaskRCNN_ResNet50_FPN_Weights,
    _utils as det_utils,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.roi_heads import maskrcnn_loss as tv_maskrcnn_loss
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RegionProposalNetwork,
    RPNHead,
)
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    import albumentations as A
except Exception:
    A = None


DEFAULT_AREA_STATS = {
    "1": {"p01": 222.63999938964844, "p99": 1963.0400390625, "median": 687.0},
    "2": {"p01": 116.0, "p99": 577.0, "median": 275.0},
    "3": {"p01": 255.69000244140625, "p99": 1027.75, "median": 571.5},
    "4": {"p01": 598.9600219726562, "p99": 14854.4404296875, "median": 2007.0},
}

CATEGORY_NAMES = {1: "class1", 2: "class2", 3: "class3", 4: "class4"}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def collate_fn(batch):
    return tuple(zip(*batch))


def read_tif_image(path: str) -> np.ndarray:
    img = tifffile.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3:
        if img.shape[0] in [1, 3, 4] and img.shape[-1] not in [1, 3, 4]:
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] >= 4:
            img = img[..., :3]
    img = img.astype(np.float32)
    if img.max() <= 255.0:
        img = img / 255.0
    else:
        p1, p99 = np.percentile(img, (1, 99))
        img = np.clip((img - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
    return img.astype(np.float32)


def polygon_or_rle_to_mask(ann: Dict, height: int, width: int) -> np.ndarray:
    seg = ann.get("segmentation", None)
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    if seg is None:
        return binary_mask
    if isinstance(seg, list):
        for poly in seg:
            if len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            cv2.fillPoly(binary_mask, [np.round(pts).astype(np.int32)], 1)
    elif isinstance(seg, dict):
        decoded = coco_mask.decode(seg)
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        binary_mask = decoded.astype(np.uint8)
    return binary_mask


def bbox_from_mask(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def mask_to_rle(binary_mask):
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = coco_mask.encode(binary_mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def masks_to_target(masks, labels, image_id, height, width):
    valid_masks, valid_labels, boxes, areas = [], [], [], []
    for mask, label in zip(masks, labels):
        mask = (mask > 0).astype(np.uint8)
        if mask.sum() <= 0:
            continue
        box = bbox_from_mask(mask)
        if box is None:
            continue
        valid_masks.append(mask)
        valid_labels.append(int(label))
        boxes.append(box)
        areas.append(float(mask.sum()))

    if len(valid_masks) == 0:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "masks": torch.zeros((0, height, width), dtype=torch.uint8),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((0,), dtype=torch.int64),
        }
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(valid_labels, dtype=torch.int64),
        "masks": torch.tensor(np.stack(valid_masks), dtype=torch.uint8),
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "area": torch.tensor(areas, dtype=torch.float32),
        "iscrowd": torch.zeros((len(valid_masks),), dtype=torch.int64),
    }


def plot_training_curves(log_path, save_path):
    if not os.path.exists(log_path):
        return
    epochs, train_losses, val_losses, val_aps = [], [], [], []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            epochs.append(rec["epoch"])
            train_losses.append(rec.get("train_loss"))
            val_losses.append(rec.get("val_loss"))
            val_aps.append(rec.get("val_ap50"))
    if len(epochs) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(epochs, train_losses, label="train_loss", marker="o", markersize=3)
    val_ep = [e for e, v in zip(epochs, val_losses) if v is not None]
    val_ls = [v for v in val_losses if v is not None]
    if val_ep:
        ax.plot(val_ep, val_ls, label="val_loss", marker="s", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax = axes[1]
    ap_ep = [e for e, v in zip(epochs, val_aps) if v is not None]
    ap_vs = [v for v in val_aps if v is not None]
    if ap_ep:
        ax.plot(ap_ep, ap_vs, label="val_AP50", marker="s",
                color="tab:green", markersize=4)
        best = max(ap_vs)
        best_ep = ap_ep[ap_vs.index(best)]
        ax.axhline(best, ls="--", color="tab:red", alpha=0.5,
                   label=f"best={best:.4f} @ ep{best_ep}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP50 (segm)")
    ax.set_title("Validation AP50")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    vmax = cm.max() if cm.size > 0 else 1
    if vmax <= 0:
        vmax = 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > 0.5 * vmax else "black", fontsize=9)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def update_confusion_matrix(cm, preds, gts, iou_thresh=0.5):
    preds_sorted = sorted(preds, key=lambda x: -x["score"])
    gt_used = [False] * len(gts)
    for p in preds_sorted:
        best_iou, best_idx = 0.0, -1
        for i, g in enumerate(gts):
            if gt_used[i]:
                continue
            iou = mask_iou(p["mask"], g["mask"])
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_idx >= 0 and best_iou >= iou_thresh:
            gt_used[best_idx] = True
            cm[gts[best_idx]["label"], p["label"]] += 1
        else:
            cm[0, p["label"]] += 1
    for i, used in enumerate(gt_used):
        if not used:
            cm[gts[i]["label"], 0] += 1


def random_color_augment(img):
    out = img.copy()
    if random.random() < 0.6:
        out = out * random.uniform(0.75, 1.25)
    if random.random() < 0.6:
        mean = out.mean(axis=(0, 1), keepdims=True)
        out = (out - mean) * random.uniform(0.75, 1.35) + mean
    if random.random() < 0.4:
        out = np.power(np.clip(out, 0.0, 1.0), random.uniform(0.7, 1.5))
    if random.random() < 0.35:
        out = out + np.random.normal(
            0, random.uniform(0.005, 0.03), out.shape
        ).astype(np.float32)
    if random.random() < 0.25:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def random_flip_rotate(img, masks):
    out_img, out_masks = img, masks
    if random.random() < 0.5:
        out_img = np.ascontiguousarray(np.flip(out_img, axis=1))
        out_masks = [np.ascontiguousarray(np.flip(m, axis=1)) for m in out_masks]
    if random.random() < 0.5:
        out_img = np.ascontiguousarray(np.flip(out_img, axis=0))
        out_masks = [np.ascontiguousarray(np.flip(m, axis=0)) for m in out_masks]
    if random.random() < 0.5:
        k = random.choice([1, 2, 3])
        out_img = np.ascontiguousarray(np.rot90(out_img, k=k))
        out_masks = [np.ascontiguousarray(np.rot90(m, k=k)) for m in out_masks]
    return out_img, out_masks


def pad_to_min_size(img, masks, min_h, min_w):
    h, w = img.shape[:2]
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return img, masks
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)),
                 mode="constant", constant_values=0)
    masks = [
        np.pad(m, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        for m in masks
    ]
    return img, masks


def random_crop_with_instances(img, masks, labels, crop_size=1024,
                               min_area=8, max_trials=30):
    img, masks = pad_to_min_size(img, masks, crop_size, crop_size)
    h, w = img.shape[:2]
    if h == crop_size and w == crop_size:
        return img, masks, labels
    best_x = random.randint(0, w - crop_size)
    best_y = random.randint(0, h - crop_size)
    for _ in range(max_trials):
        x = random.randint(0, w - crop_size)
        y = random.randint(0, h - crop_size)
        if any(m[y:y + crop_size, x:x + crop_size].sum() >= min_area for m in masks):
            best_x, best_y = x, y
            break
    x, y = best_x, best_y
    crop_img = img[y:y + crop_size, x:x + crop_size]
    crop_masks, crop_labels = [], []
    for m, lab in zip(masks, labels):
        cm = m[y:y + crop_size, x:x + crop_size].astype(np.uint8)
        if cm.sum() < min_area:
            continue
        crop_masks.append(cm)
        crop_labels.append(lab)
    return crop_img, crop_masks, crop_labels


def random_resize(img, masks, base=1024, ratio_range=(0.5, 2.0)):
    h, w = img.shape[:2]
    r = random.uniform(*ratio_range)
    target = int(round(base * r))
    scale = target / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    masks_r = [
        cv2.resize(m.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
        for m in masks
    ]
    return img_r, masks_r


def elastic_deform(img, masks, alpha=40.0, sigma=8.0):
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur(
        (np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma
    ) * alpha
    dy = cv2.GaussianBlur(
        (np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma
    ) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    img_d = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT)
    masks_d = [
        cv2.remap(m, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                  borderMode=cv2.BORDER_CONSTANT)
        for m in masks
    ]
    return img_d, masks_d


def build_albu_transform():
    if A is None:
        return None
    return A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.08, scale_limit=0.15, rotate_limit=25,
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.65,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.18, contrast_limit=0.18, p=0.45,
        ),
        A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32, min_holes=1,
            min_height=8, min_width=8, fill_value=0, mask_fill_value=0, p=0.25,
        ),
    ])


class CellDataset(Dataset):
    def __init__(self, img_dir, ann_json_path, is_train=False,
                 use_copy_paste=False, use_random_crop=False, crop_size=1024,
                 copy_paste_prob=0.5, max_paste_objects=8, use_albu=False,
                 use_random_resize=False, use_elastic=False):
        self.img_dir = img_dir
        self.ann_json_path = ann_json_path
        self.is_train = is_train
        self.use_copy_paste = use_copy_paste
        self.use_random_crop = use_random_crop
        self.crop_size = crop_size
        self.copy_paste_prob = copy_paste_prob
        self.max_paste_objects = max_paste_objects
        self.use_albu = use_albu
        self.use_random_resize = use_random_resize
        self.use_elastic = use_elastic
        self.albu_transform = build_albu_transform() if (use_albu and is_train) else None
        with open(ann_json_path, "r", encoding="utf-8") as f:
            self.coco = json.load(f)
        self.images = self.coco["images"]
        self.ann_map = {}
        for ann in self.coco.get("annotations", []):
            self.ann_map.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def load_raw(self, idx):
        img_info = self.images[idx]
        img_id = int(img_info["id"])
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = read_tif_image(img_path)
        h, w = img.shape[:2]
        anns = self.ann_map.get(img_id, [])
        masks, labels = [], []
        for ann in anns:
            mask = polygon_or_rle_to_mask(ann, h, w)
            if mask.sum() <= 0:
                continue
            label = int(ann["category_id"])
            if label < 1 or label > 4:
                continue
            masks.append(mask.astype(np.uint8))
            labels.append(label)
        return img, masks, labels, img_id

    def copy_paste_augment(self, img, masks, labels):
        if len(self.images) <= 1:
            return img, masks, labels
        h, w = img.shape[:2]
        out_img = img.copy()
        out_masks = list(masks)
        out_labels = list(labels)
        union = np.zeros((h, w), dtype=np.uint8)
        for m in out_masks:
            union = np.maximum(union, (m > 0).astype(np.uint8))
        for _ in range(random.randint(1, self.max_paste_objects)):
            src_idx = random.randint(0, len(self.images) - 1)
            src_img, src_masks, src_labels, _ = self.load_raw(src_idx)
            if not src_masks:
                continue
            j = random.randint(0, len(src_masks) - 1)
            src_mask = src_masks[j].astype(np.uint8)
            src_label = int(src_labels[j])
            box = bbox_from_mask(src_mask)
            if box is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            patch_img = src_img[y1:y2, x1:x2]
            patch_mask = src_mask[y1:y2, x1:x2]
            ph, pw = patch_mask.shape[:2]
            if ph < 3 or pw < 3 or ph >= h or pw >= w:
                continue
            dx = random.randint(0, w - pw)
            dy = random.randint(0, h - ph)
            new_mask = np.zeros((h, w), dtype=np.uint8)
            new_mask[dy:dy + ph, dx:dx + pw] = patch_mask
            overlap = np.logical_and(new_mask > 0, union > 0).sum()
            if overlap / max(1, new_mask.sum()) > 0.3:
                continue
            roi = out_img[dy:dy + ph, dx:dx + pw]
            m_bool = patch_mask.astype(bool)
            if roi.shape[:2] != patch_mask.shape[:2]:
                continue
            roi[m_bool] = patch_img[m_bool]
            out_img[dy:dy + ph, dx:dx + pw] = roi
            out_masks.append(new_mask)
            out_labels.append(src_label)
            union = np.maximum(union, new_mask)
        return out_img, out_masks, out_labels

    def __getitem__(self, idx):
        img, masks, labels, img_id = self.load_raw(idx)
        if self.is_train:
            if self.use_copy_paste and random.random() < self.copy_paste_prob:
                img, masks, labels = self.copy_paste_augment(img, masks, labels)
            if self.use_random_resize:
                img, masks = random_resize(
                    img, masks, base=self.crop_size, ratio_range=(0.5, 2.0),
                )
            if self.use_random_crop:
                img, masks, labels = random_crop_with_instances(
                    img, masks, labels, crop_size=self.crop_size,
                )
            if self.albu_transform is not None and len(masks) > 0:
                aug = self.albu_transform(image=img, masks=masks)
                img = aug["image"].astype(np.float32)
                masks = [m.astype(np.uint8) for m in aug["masks"]]
            img, masks = random_flip_rotate(img, masks)
            if self.use_elastic and random.random() < 0.1 and len(masks) > 0:
                img, masks = elastic_deform(img, masks)
            img = random_color_augment(img)
        h, w = img.shape[:2]
        target = masks_to_target(masks, labels, img_id, h, w)
        return torch.from_numpy(img).permute(2, 0, 1).float(), target


class ConvNeXtV2FPNBackbone(nn.Module):
    def __init__(self, model_name="convnextv2_base.fcmae_ft_in22k_in1k",
                 drop_path_rate=0.1, freeze_stem=False):
        super().__init__()
        if not _HAS_TIMM:
            raise RuntimeError("Please `pip install timm` to use ConvNeXt-V2 backbone.")
        self.body = timm.create_model(
            model_name, pretrained=True, features_only=True,
            out_indices=(0, 1, 2, 3), drop_path_rate=drop_path_rate,
        )
        in_channels_list = self.body.feature_info.channels()
        if freeze_stem:
            for name, p in self.body.named_parameters():
                if "stem" in name:
                    p.requires_grad = False
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=256,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = 256

    def forward(self, x):
        feats = self.body(x)
        od = OrderedDict()
        for i, f in enumerate(feats):
            od[str(i)] = f
        return self.fpn(od)


class CascadeRoIHeads(nn.Module):
    def __init__(self,
                 box_roi_pool: MultiScaleRoIAlign,
                 box_head: nn.ModuleList,
                 box_predictor: nn.ModuleList,
                 iou_thresholds=(0.5, 0.6, 0.7),
                 bbox_reg_weights=((10., 10., 5., 5.),
                                   (20., 20., 10., 10.),
                                   (30., 30., 15., 15.)),
                 stage_loss_weights=(1.0, 0.5, 0.25),
                 batch_size_per_image=512,
                 positive_fraction=0.25,
                 score_thresh=0.05,
                 nms_thresh=0.5,
                 detections_per_img=300,
                 mask_roi_pool: MultiScaleRoIAlign = None,
                 mask_head: nn.Module = None,
                 mask_predictor: nn.Module = None):
        super().__init__()
        assert len(box_head) == len(box_predictor) == len(iou_thresholds) == 3
        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.num_stages = len(iou_thresholds)
        self.iou_thresholds = iou_thresholds
        self.bbox_reg_weights = bbox_reg_weights
        self.stage_loss_weights = stage_loss_weights

        self.proposal_matchers = nn.ModuleList()
        self.fg_bg_samplers = nn.ModuleList()
        self.box_coders = []
        self._matchers = []
        self._samplers = []
        for thr, w in zip(iou_thresholds, bbox_reg_weights):
            self._matchers.append(
                det_utils.Matcher(thr, thr, allow_low_quality_matches=False)
            )
            self._samplers.append(
                det_utils.BalancedPositiveNegativeSampler(
                    batch_size_per_image, positive_fraction,
                )
            )
            self.box_coders.append(det_utils.BoxCoder(weights=w))

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

    def _assign_targets_and_sample(self, proposals, targets, stage):
        matched_idxs_list, labels_list = [], []
        sampled_proposals_list, regression_targets_list = [], []
        matcher = self._matchers[stage]
        sampler = self._samplers[stage]
        box_coder = self.box_coders[stage]

        for props, tgt in zip(proposals, targets):
            gt_boxes = tgt["boxes"].to(props.device)
            gt_labels = tgt["labels"].to(props.device)

            if gt_boxes.numel() == 0:
                device = props.device
                matched_idxs = torch.zeros(
                    (props.shape[0],), dtype=torch.int64, device=device,
                )
                labels = torch.zeros(
                    (props.shape[0],), dtype=torch.int64, device=device,
                )
                matched_gt_boxes = torch.zeros_like(props)
            else:
                iou = box_ops.box_iou(gt_boxes, props)
                matched_idxs = matcher(iou)
                clamped = matched_idxs.clamp(min=0)
                matched_gt_boxes = gt_boxes[clamped]
                labels = gt_labels[clamped]
                labels[matched_idxs == matcher.BELOW_LOW_THRESHOLD] = 0
                labels[matched_idxs == matcher.BETWEEN_THRESHOLDS] = -1

            sampled_pos_inds, sampled_neg_inds = sampler([labels])
            sampled_pos_inds = torch.where(sampled_pos_inds[0])[0]
            sampled_neg_inds = torch.where(sampled_neg_inds[0])[0]
            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

            sampled_props = props[sampled_inds]
            sampled_labels = labels[sampled_inds]
            sampled_gt_boxes = matched_gt_boxes[sampled_inds]

            reg_targets = box_coder.encode_single(sampled_gt_boxes, sampled_props)

            matched_idxs_list.append(matched_idxs[sampled_inds])
            labels_list.append(sampled_labels)
            sampled_proposals_list.append(sampled_props)
            regression_targets_list.append(reg_targets)

        return (
            sampled_proposals_list,
            labels_list,
            regression_targets_list,
            matched_idxs_list,
        )

    def _stage_forward(self, features, proposals, image_shapes, stage):
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head[stage](box_features)
        class_logits, box_regression = self.box_predictor[stage](box_features)
        return class_logits, box_regression

    def _refine_boxes(self, proposals, box_regression, class_logits,
                      image_shapes, stage):
        box_coder = self.box_coders[stage]
        num_classes = class_logits.shape[-1]

        device = box_regression.device
        all_refined = []
        offset = 0
        for props in proposals:
            n = props.shape[0]
            reg = box_regression[offset:offset + n].reshape(n, num_classes, 4)
            logits = class_logits[offset:offset + n]
            scores = F.softmax(logits, dim=-1)
            if num_classes > 1:
                fg_scores = scores[:, 1:]
                pred_cls = fg_scores.argmax(dim=-1) + 1
            else:
                pred_cls = torch.zeros(n, dtype=torch.long, device=device)
            pred_reg = reg[torch.arange(n, device=device), pred_cls]
            refined = box_coder.decode_single(pred_reg, props)
            refined = box_ops.clip_boxes_to_image(
                refined, image_shapes[len(all_refined)],
            )
            all_refined.append(refined)
            offset += n
        return all_refined

    def _stage_loss(self, class_logits, box_regression, labels, regression_targets):
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        classification_loss = F.cross_entropy(class_logits, labels.clamp(min=0))

        sampled_pos_inds = torch.where(labels > 0)[0]
        if sampled_pos_inds.numel() == 0:
            box_loss = box_regression.sum() * 0.0
        else:
            labels_pos = labels[sampled_pos_inds]
            N, num_classes_x4 = box_regression.shape
            num_classes = num_classes_x4 // 4
            box_regression = box_regression.reshape(N, num_classes, 4)
            box_loss = F.smooth_l1_loss(
                box_regression[sampled_pos_inds, labels_pos],
                regression_targets[sampled_pos_inds],
                beta=1.0 / 9, reduction="sum",
            ) / labels.numel()
        return classification_loss, box_loss

    def _postprocess_detections(self, class_logits_list, box_regression_list,
                                proposals, image_shapes):
        device = class_logits_list[0].device
        box_coder = self.box_coders[-1]
        num_classes = class_logits_list[-1].shape[-1]

        scores_per_stage = [F.softmax(cl, dim=-1) for cl in class_logits_list]
        avg_scores = torch.stack(scores_per_stage, dim=0).mean(dim=0)

        final_reg = box_regression_list[-1]
        result_boxes, result_scores, result_labels = [], [], []
        offset = 0
        for props, img_shape in zip(proposals, image_shapes):
            n = props.shape[0]
            sc = avg_scores[offset:offset + n]
            reg = final_reg[offset:offset + n].reshape(n, num_classes, 4)
            boxes_per_cls = []
            for c in range(num_classes):
                boxes_per_cls.append(box_coder.decode_single(reg[:, c], props))
            boxes_per_cls = torch.stack(boxes_per_cls, dim=1)
            boxes_per_cls = box_ops.clip_boxes_to_image(
                boxes_per_cls.reshape(-1, 4), img_shape,
            ).reshape(n, num_classes, 4)

            scores = sc[:, 1:]
            boxes = boxes_per_cls[:, 1:]
            labels = torch.arange(1, num_classes, device=device).view(1, -1).expand_as(scores)

            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            keep = scores > self.score_thresh
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            keep = box_ops.remove_small_boxes(boxes, min_size=1.0)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            result_boxes.append(boxes)
            result_scores.append(scores)
            result_labels.append(labels)
            offset += n
        return result_boxes, result_scores, result_labels

    def forward(self, features, proposals, image_shapes, targets=None):
        losses = {}

        all_class_logits, all_box_regs = [], []
        cur_proposals = proposals

        if self.training:
            for stage in range(self.num_stages):
                (
                    sampled_props,
                    labels,
                    reg_targets,
                    matched_idxs,
                ) = self._assign_targets_and_sample(cur_proposals, targets, stage)
                class_logits, box_regression = self._stage_forward(
                    features, sampled_props, image_shapes, stage,
                )
                cls_loss, box_loss = self._stage_loss(
                    class_logits, box_regression, labels, reg_targets,
                )
                w = self.stage_loss_weights[stage]
                losses[f"loss_classifier_s{stage}"] = cls_loss * w
                losses[f"loss_box_reg_s{stage}"] = box_loss * w

                if stage < self.num_stages - 1:
                    with torch.no_grad():
                        cur_proposals = self._refine_boxes(
                            sampled_props, box_regression.detach(),
                            class_logits.detach(), image_shapes, stage,
                        )
                else:
                    final_sampled_props = sampled_props
                    final_labels = labels
                    final_matched_idxs = matched_idxs
        else:
            for stage in range(self.num_stages):
                class_logits, box_regression = self._stage_forward(
                    features, cur_proposals, image_shapes, stage,
                )
                all_class_logits.append(class_logits)
                all_box_regs.append(box_regression)
                if stage < self.num_stages - 1:
                    cur_proposals = self._refine_boxes(
                        cur_proposals, box_regression, class_logits,
                        image_shapes, stage,
                    )

        result = []
        if self.training:
            pos_props_list, pos_matched_idx_list = [], []
            for props, labs, midx in zip(
                final_sampled_props, final_labels, final_matched_idxs,
            ):
                pos = torch.where(labs > 0)[0]
                if pos.numel() == 0:
                    pos_props_list.append(props.new_zeros((0, 4)))
                    pos_matched_idx_list.append(
                        midx.new_zeros((0,), dtype=torch.int64),
                    )
                else:
                    pos_props_list.append(props[pos])
                    pos_matched_idx_list.append(
                        midx[pos].clamp(min=0).to(torch.int64),
                    )

            total_pos = sum(p.shape[0] for p in pos_props_list)
            if total_pos == 0:
                any_param = next(self.mask_predictor.parameters())
                losses["loss_mask"] = any_param.sum() * 0.0
            else:
                mask_features = self.mask_roi_pool(
                    features, pos_props_list, image_shapes,
                )
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)

                gt_masks_list = [t["masks"].to(mask_logits.device) for t in targets]
                gt_labels_list = [t["labels"].to(mask_logits.device) for t in targets]

                losses["loss_mask"] = tv_maskrcnn_loss(
                    mask_logits,
                    pos_props_list,
                    gt_masks_list,
                    gt_labels_list,
                    pos_matched_idx_list,
                )
            return result, losses
        else:
            boxes, scores, labels = self._postprocess_detections(
                all_class_logits, all_box_regs, cur_proposals, image_shapes,
            )
            mask_features = self.mask_roi_pool(features, boxes, image_shapes)
            if mask_features.numel() > 0:
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
                masks_probs = []
                offset = 0
                for b, l in zip(boxes, labels):
                    n = b.shape[0]
                    if n == 0:
                        masks_probs.append(b.new_zeros((0, 1, 28, 28)))
                        continue
                    ml = mask_logits[offset:offset + n]
                    idx = torch.arange(n, device=ml.device)
                    m = ml[idx, l][:, None]
                    masks_probs.append(m.sigmoid())
                    offset += n
            else:
                masks_probs = [b.new_zeros((0, 1, 28, 28)) for b in boxes]

            for b, s, l, m in zip(boxes, scores, labels, masks_probs):
                result.append({
                    "boxes": b, "scores": s, "labels": l, "masks": m,
                })
            return result, losses


class CascadeMaskRCNN(nn.Module):
    def __init__(self, backbone, num_classes,
                 rpn_anchor_generator, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_post_nms_top_n_train=1000,
                 rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.45,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 box_roi_pool=None, box_head_dim=1024,
                 box_score_thresh=0.05, box_nms_thresh=0.5,
                 box_detections_per_img=300,
                 mask_roi_pool=None,
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 iou_thresholds=(0.5, 0.6, 0.7),
                 bbox_reg_weights=((10., 10., 5., 5.),
                                   (20., 20., 10., 10.),
                                   (30., 30., 15., 15.)),
                 stage_loss_weights=(1.0, 0.5, 0.25)):
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels

        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels,
                rpn_anchor_generator.num_anchors_per_location()[0],
            )
        rpn_pre = dict(training=rpn_pre_nms_top_n_train,
                       testing=rpn_pre_nms_top_n_test)
        rpn_post = dict(training=rpn_post_nms_top_n_train,
                        testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre, rpn_post, rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"],
                output_size=7, sampling_ratio=2,
            )
        resolution = box_roi_pool.output_size[0]
        box_heads = nn.ModuleList([
            TwoMLPHead(out_channels * resolution ** 2, box_head_dim)
            for _ in range(3)
        ])
        box_predictors = nn.ModuleList([
            FastRCNNPredictor(box_head_dim, num_classes) for _ in range(3)
        ])

        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"],
                output_size=14, sampling_ratio=2,
            )
        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)
        mask_predictor = MaskRCNNPredictor(mask_layers[-1], 256, num_classes)

        self.roi_heads = CascadeRoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=box_heads,
            box_predictor=box_predictors,
            iou_thresholds=iou_thresholds,
            bbox_reg_weights=bbox_reg_weights,
            stage_loss_weights=stage_loss_weights,
            score_thresh=box_score_thresh,
            nms_thresh=box_nms_thresh,
            detections_per_img=box_detections_per_img,
            mask_roi_pool=mask_roi_pool,
            mask_head=mask_head,
            mask_predictor=mask_predictor,
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(
            min_size, max_size, image_mean, image_std,
        )

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("Targets required in training.")
        original_image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets,
        )
        if not self.training and len(detections) > 0:
            detections = self._paste_masks(
                detections, images.image_sizes, original_image_sizes,
            )
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if self.training:
            return losses
        return detections

    @staticmethod
    def _paste_masks(detections, image_sizes, original_image_sizes):
        from torchvision.models.detection.roi_heads import paste_masks_in_image
        results = []
        for det, im_s, orig_s in zip(
            detections, image_sizes, original_image_sizes,
        ):
            if det["masks"].shape[0] > 0:
                masks = paste_masks_in_image(det["masks"], det["boxes"], im_s)
                if im_s != orig_s:
                    masks = F.interpolate(
                        masks.float(), size=orig_s,
                        mode="bilinear", align_corners=False,
                    )
                    sx = orig_s[1] / im_s[1]
                    sy = orig_s[0] / im_s[0]
                    boxes = det["boxes"].clone()
                    boxes[:, [0, 2]] *= sx
                    boxes[:, [1, 3]] *= sy
                    det["boxes"] = boxes
                det["masks"] = masks
            results.append(det)
        return results


def get_anchor_sizes(anchor_preset):
    if anchor_preset == "youzhe":
        return ((16,), (32,), (64,), (128,), (256,))
    if anchor_preset == "default":
        return ((4,), (8,), (16,), (32,), (64,))
    if anchor_preset == "area_light":
        return ((8,), (16,), (32,), (64,), (128,))
    if anchor_preset == "area":
        return ((8, 12), (16, 24), (32, 48), (64, 96), (128, 160))
    raise ValueError(f"Unknown anchor_preset: {anchor_preset}")


def get_anchor_ratios(anchor_preset):
    if anchor_preset == "youzhe":
        return ((1.0,),) * 5
    sizes = get_anchor_sizes(anchor_preset)
    return ((0.5, 1.0, 2.0),) * len(sizes)


def build_model(num_classes=5,
                detector="cascade_mask_rcnn",
                backbone="convnextv2_base",
                use_coco_pretrained=False,
                box_score_thresh=0.05,
                box_nms_thresh=0.5,
                detections_per_img=300,
                anchor_preset="youzhe",
                drop_path_rate=0.1,
                rpn_nms_thresh=0.45,
                min_size=1024, max_size=1024):
    anchor_sizes = get_anchor_sizes(anchor_preset)
    anchor_ratios = get_anchor_ratios(anchor_preset)
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes, aspect_ratios=anchor_ratios,
    )
    print(
        f"[build_model] detector={detector}, backbone={backbone}, "
        f"anchor_preset={anchor_preset}, sizes={anchor_sizes}, "
        f"ratios={anchor_ratios}"
    )

    if detector == "cascade_mask_rcnn":
        if backbone != "convnextv2_base":
            raise ValueError(
                "Cascade path currently wired to convnextv2_base only."
            )
        fpn_backbone = ConvNeXtV2FPNBackbone(drop_path_rate=drop_path_rate)
        return CascadeMaskRCNN(
            backbone=fpn_backbone, num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            rpn_nms_thresh=rpn_nms_thresh,
            rpn_pre_nms_top_n_train=2000, rpn_post_nms_top_n_train=1000,
            rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_test=1000,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=detections_per_img,
            min_size=min_size, max_size=max_size,
        )

    if detector == "mask_rcnn":
        if use_coco_pretrained:
            model = maskrcnn_resnet50_fpn(
                weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                rpn_anchor_generator=anchor_generator,
                box_score_thresh=box_score_thresh,
                box_nms_thresh=box_nms_thresh,
                box_detections_per_img=detections_per_img,
                rpn_nms_thresh=rpn_nms_thresh,
                min_size=min_size, max_size=max_size,
            )
            in_f = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
            in_fm = model.roi_heads.mask_predictor.conv5_mask.in_channels
            model.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_fm, 256, num_classes,
            )
        else:
            model = maskrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
                num_classes=num_classes,
                rpn_anchor_generator=anchor_generator,
                box_score_thresh=box_score_thresh,
                box_nms_thresh=box_nms_thresh,
                box_detections_per_img=detections_per_img,
                rpn_nms_thresh=rpn_nms_thresh,
                min_size=min_size, max_size=max_size,
            )
        return model
    raise ValueError(f"Unknown detector: {detector}")


def mask_iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def compute_area_stats(dataset):
    areas = {1: [], 2: [], 3: [], 4: []}
    for idx in tqdm(range(len(dataset)), desc="Compute area stats"):
        _, masks, labels, _ = dataset.load_raw(idx)
        for m, lab in zip(masks, labels):
            areas[int(lab)].append(float(m.sum()))
    stats = {}
    for cls_id in [1, 2, 3, 4]:
        arr = np.array(areas[cls_id], dtype=np.float32)
        if len(arr) == 0:
            stats[str(cls_id)] = {"p01": 0.0, "p99": 1e12, "median": 0.0}
        else:
            stats[str(cls_id)] = {
                "p01": float(np.percentile(arr, 1)),
                "p99": float(np.percentile(arr, 99)),
                "median": float(np.median(arr)),
            }
    return stats


def area_filter_ok(mask, label, area_stats, min_ratio=0.35, max_ratio=2.5):
    if area_stats is None:
        return True
    cs = area_stats.get(str(int(label)), None)
    if cs is None:
        return True
    area = float(mask.sum())
    if area < max(2.0, cs["p01"] * min_ratio):
        return False
    if area > cs["p99"] * max_ratio:
        return False
    return True


def postprocess_prediction_dict(pred, score_thresh=0.05, mask_thresh=0.5,
                                mask_nms_thresh=0.5, max_per_img=300,
                                area_stats=None):
    boxes = pred["boxes"]
    labels = pred["labels"]
    scores = pred["scores"]
    masks_prob = pred["masks"]
    candidates = []
    for box, label, score, mp in zip(boxes, labels, scores, masks_prob):
        if float(score) < score_thresh:
            continue
        label = int(label)
        if label < 1 or label > 4:
            continue
        binary = (mp >= mask_thresh).astype(np.uint8)
        if binary.sum() <= 0:
            continue
        num, cc, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8,
        )
        if num > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            binary = (cc == largest).astype(np.uint8)
        if binary.sum() <= 0:
            continue
        if not area_filter_ok(binary, label, area_stats):
            continue
        rb = bbox_from_mask(binary)
        if rb is None:
            continue
        candidates.append({
            "box": rb, "label": label,
            "score": float(score), "mask": binary,
        })
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    kept = []
    for c in candidates:
        if any(
            c["label"] == k["label"]
            and mask_iou(c["mask"], k["mask"]) > mask_nms_thresh
            for k in kept
        ):
            continue
        kept.append(c)
        if len(kept) >= max_per_img:
            break
    return kept


@torch.no_grad()
def predict_single_image(model, img_tensor, device, use_tta=False):
    model.eval()
    _, h, w = img_tensor.shape
    all_b, all_l, all_s, all_m = [], [], [], []

    def run_one(x):
        out = model([x.to(device)])[0]
        return (
            out["boxes"].cpu().numpy(),
            out["labels"].cpu().numpy(),
            out["scores"].cpu().numpy(),
            out["masks"].cpu().numpy()[:, 0],
        )

    b, l, s, m = run_one(img_tensor)
    all_b.append(b)
    all_l.append(l)
    all_s.append(s)
    all_m.append(m)

    if use_tta:
        x = torch.flip(img_tensor, dims=[2])
        b, l, s, m = run_one(x)
        if len(b) > 0:
            b[:, [0, 2]] = w - b[:, [2, 0]]
            m = np.flip(m, axis=2).copy()
        all_b.append(b)
        all_l.append(l)
        all_s.append(s)
        all_m.append(m)

        x = torch.flip(img_tensor, dims=[1])
        b, l, s, m = run_one(x)
        if len(b) > 0:
            b[:, [1, 3]] = h - b[:, [3, 1]]
            m = np.flip(m, axis=1).copy()
        all_b.append(b)
        all_l.append(l)
        all_s.append(s)
        all_m.append(m)

        x = torch.flip(img_tensor, dims=[1, 2])
        b, l, s, m = run_one(x)
        if len(b) > 0:
            b[:, [0, 2]] = w - b[:, [2, 0]]
            b[:, [1, 3]] = h - b[:, [3, 1]]
            m = np.flip(np.flip(m, axis=1), axis=2).copy()
        all_b.append(b)
        all_l.append(l)
        all_s.append(s)
        all_m.append(m)

    boxes = np.concatenate(all_b, axis=0) if all_b else np.zeros((0, 4))
    labels = np.concatenate(all_l, axis=0) if all_l else np.zeros((0,))
    scores = np.concatenate(all_s, axis=0) if all_s else np.zeros((0,))
    masks = np.concatenate(all_m, axis=0) if all_m else np.zeros((0, h, w))
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)
    return {
        "boxes": boxes.astype(np.float32),
        "labels": labels.astype(np.int64),
        "scores": scores.astype(np.float32),
        "masks": masks.astype(np.float32),
    }


@torch.no_grad()
def evaluate_ap50(model, dataset, loader, device,
                  score_thresh=0.05, mask_thresh=0.5, mask_nms_thresh=0.5,
                  max_per_img=300, use_tta=False, area_stats=None,
                  num_classes=4, cm_iou_thresh=0.5, compute_cm=False):
    model.eval()
    results = []
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    id_to_idx = {int(im["id"]): i for i, im in enumerate(dataset.images)}
    for imgs, targets in tqdm(loader, desc="Val AP50", leave=False):
        for img_tensor, target in zip(imgs, targets):
            image_id = int(target["image_id"].item())
            raw = predict_single_image(
                model, img_tensor, device, use_tta=use_tta,
            )
            final = postprocess_prediction_dict(
                raw, score_thresh, mask_thresh,
                mask_nms_thresh, max_per_img, area_stats,
            )
            for p in final:
                results.append({
                    "image_id": image_id,
                    "category_id": int(p["label"]),
                    "bbox": xyxy_to_xywh(p["box"]),
                    "segmentation": mask_to_rle(p["mask"]),
                    "score": float(p["score"]),
                })
            if compute_cm and image_id in id_to_idx:
                _, gt_masks, gt_labels, _ = dataset.load_raw(id_to_idx[image_id])
                gts = [
                    {"label": int(l), "mask": m}
                    for m, l in zip(gt_masks, gt_labels)
                ]
                update_confusion_matrix(cm, final, gts, iou_thresh=cm_iou_thresh)
    if not results:
        return 0.0, cm
    coco_gt = COCO(dataset.ann_json_path)
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.params.imgIds = [int(x["id"]) for x in dataset.images]
    coco_eval.params.catIds = [1, 2, 3, 4]
    coco_eval.params.maxDets = [1, 10, max_per_img]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[1]), cm


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.train()
    total, count = 0.0, 0
    for imgs, targets in tqdm(loader, desc="Val Loss", leave=False):
        imgs = [i.to(device) for i in imgs]
        targets = [
            {k: v.to(device) for k, v in t.items()} for t in targets
        ]
        loss_dict = model(imgs, targets)
        total += float(sum(loss_dict.values()).item())
        count += 1
    return total / max(1, count)


def train_one_epoch(model, loader, optimizer, device,
                    scaler=None, use_amp=False, grad_clip=1.0,
                    lr_scheduler_warmup=None, warmup_iters=0, global_step=0):
    model.train()
    total, count = 0.0, 0
    for imgs, targets in tqdm(loader, desc="Train", leave=False):
        imgs = [i.to(device) for i in imgs]
        targets = [
            {k: v.to(device) for k, v in t.items()} for t in targets
        ]
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if lr_scheduler_warmup is not None and global_step < warmup_iters:
            lr_scheduler_warmup.step()
        total += float(loss.item())
        count += 1
        global_step += 1
    return total / max(1, count), global_step


def save_checkpoint(model, optimizer, scheduler, epoch, best_ap50, path):
    torch.save({
        "epoch": epoch,
        "best_ap50": best_ap50,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }, path)


def load_model_weights(model, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(sd)
    return model


def build_optimizer(model, args):
    backbone_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone.body"):
            backbone_params.append(p)
        else:
            other_params.append(p)
    param_groups = [
        {"params": backbone_params, "lr": args.lr * 0.1,
         "weight_decay": args.weight_decay},
        {"params": other_params, "lr": args.lr,
         "weight_decay": args.weight_decay},
    ]
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            param_groups, lr=args.lr, weight_decay=args.weight_decay,
        )
    elif args.optimizer == "sgd":
        return torch.optim.SGD(
            param_groups, lr=args.lr, momentum=0.9,
            weight_decay=args.weight_decay,
        )
    raise ValueError(args.optimizer)


def train(args):
    ensure_dir(args.output_dir)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_json = os.path.join(args.ann_dir, "train.json")
    val_json = os.path.join(args.ann_dir, "val.json")

    train_ds = CellDataset(
        img_dir=args.train_dir, ann_json_path=train_json, is_train=True,
        use_copy_paste=args.use_copy_paste, use_random_crop=args.use_random_crop,
        crop_size=args.crop_size, copy_paste_prob=args.copy_paste_prob,
        max_paste_objects=args.max_paste_objects, use_albu=args.use_albu,
        use_random_resize=args.use_random_resize, use_elastic=args.use_elastic,
    )
    val_ds = CellDataset(
        img_dir=args.train_dir, ann_json_path=val_json, is_train=False,
    )

    area_stats = compute_area_stats(train_ds)
    with open(
        os.path.join(args.output_dir, "area_stats.json"), "w", encoding="utf-8",
    ) as f:
        json.dump(area_stats, f, indent=2)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    model = build_model(
        num_classes=args.num_classes, detector=args.detector,
        backbone=args.backbone, use_coco_pretrained=args.use_coco_pretrained,
        box_score_thresh=args.model_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        detections_per_img=args.max_per_img,
        anchor_preset=args.anchor_preset,
        drop_path_rate=args.drop_path_rate,
        rpn_nms_thresh=args.rpn_nms_thresh,
        min_size=args.crop_size, max_size=args.crop_size,
    ).to(device)

    optimizer = build_optimizer(model, args)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=args.lr * 0.01,
    )
    warmup_iters = args.warmup_epochs * len(train_loader)
    warmup_scheduler = (
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0 / max(1, warmup_iters),
            end_factor=1.0, total_iters=max(1, warmup_iters),
        )
        if warmup_iters > 0
        else None
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    best_ap50 = -1.0
    global_step = 0
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    open(log_path, "w", encoding="utf-8").close()
    cm_class_names = ["background"] + [CATEGORY_NAMES[i] for i in [1, 2, 3, 4]]

    for epoch in range(1, args.epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.epochs} ==========")
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, device, scaler,
            args.use_amp, args.grad_clip,
            lr_scheduler_warmup=warmup_scheduler,
            warmup_iters=warmup_iters, global_step=global_step,
        )
        if epoch > args.warmup_epochs:
            main_scheduler.step()

        do_val = (
            (epoch % args.val_interval == 0)
            or (epoch == args.epochs)
            or (args.val_at_epoch1 and epoch == 1)
        )
        val_loss, val_ap50, cm = None, None, None
        if do_val:
            if args.compute_val_loss:
                val_loss = evaluate_loss(model, val_loader, device)
            val_ap50, cm = evaluate_ap50(
                model, val_ds, val_loader, device,
                score_thresh=args.score_thresh, mask_thresh=args.mask_thresh,
                mask_nms_thresh=args.mask_nms_thresh,
                max_per_img=args.max_per_img,
                use_tta=False, area_stats=None, num_classes=4,
                cm_iou_thresh=0.5, compute_cm=args.train_compute_cm,
            )
            if args.train_compute_cm and cm is not None:
                plot_confusion_matrix(
                    cm, cm_class_names,
                    os.path.join(args.output_dir, "confusion_matrix_latest.png"),
                    title=f"CM (epoch {epoch}, AP50={val_ap50:.4f})",
                )

        val_ls = f"{val_loss:.4f}" if val_loss is not None else "NA"
        val_as = f"{val_ap50:.4f}" if val_ap50 is not None else "NA"
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_ls}, "
            f"val_AP50={val_as}, lr={optimizer.param_groups[1]['lr']:.6g}"
        )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_ap50": val_ap50,
                "lr": optimizer.param_groups[1]["lr"],
            }) + "\n")
        plot_training_curves(
            log_path, os.path.join(args.output_dir, "training_curves.png"),
        )

        save_checkpoint(
            model, optimizer, main_scheduler, epoch, best_ap50,
            os.path.join(args.output_dir, "last.pth"),
        )
        if val_ap50 is not None and val_ap50 > best_ap50:
            best_ap50 = val_ap50
            save_checkpoint(
                model, optimizer, main_scheduler, epoch, best_ap50,
                os.path.join(args.output_dir, "best_ap50.pth"),
            )
            if args.train_compute_cm and cm is not None:
                plot_confusion_matrix(
                    cm, cm_class_names,
                    os.path.join(args.output_dir, "confusion_matrix_best.png"),
                    title=f"CM (best ep{epoch}, AP50={val_ap50:.4f})",
                )
            print(f"Saved best AP50: {best_ap50:.4f}")

    print(f"\nDone. Best AP50 = {best_ap50:.4f}")


def load_test_id_mapping(test_ids_json):
    with open(test_ids_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {item["file_name"]: int(item["id"]) for item in data}
    if isinstance(data, dict):
        return {k: int(v) for k, v in data.items()}
    raise ValueError("Unsupported test_image_name_to_ids.json format.")


def inference(args):
    ensure_dir(args.output_dir)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    area_stats = None
    asp = os.path.join(args.output_dir, "area_stats.json")
    if args.use_area_filter and os.path.exists(asp):
        with open(asp, "r", encoding="utf-8") as f:
            area_stats = json.load(f)
    elif args.use_area_filter:
        area_stats = DEFAULT_AREA_STATS

    name_to_id = load_test_id_mapping(args.test_ids_json)

    model = build_model(
        num_classes=args.num_classes, detector=args.detector,
        backbone=args.backbone, use_coco_pretrained=args.use_coco_pretrained,
        box_score_thresh=args.model_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        detections_per_img=args.max_per_img,
        anchor_preset=args.anchor_preset,
        drop_path_rate=args.drop_path_rate,
        rpn_nms_thresh=args.rpn_nms_thresh,
        min_size=args.crop_size, max_size=args.crop_size,
    )
    model = load_model_weights(model, args.checkpoint, device).to(device).eval()

    test_files = sorted([
        f for f in os.listdir(args.test_dir)
        if f.lower().endswith((".tif", ".tiff"))
    ])
    results = []
    for fname in tqdm(test_files, desc="Inference"):
        if fname not in name_to_id:
            print(f"Warning: {fname} not in id mapping. Skip.")
            continue
        image_id = int(name_to_id[fname])
        img = read_tif_image(os.path.join(args.test_dir, fname))
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        raw = predict_single_image(model, img_t, device, use_tta=args.use_tta)
        final = postprocess_prediction_dict(
            raw, args.score_thresh, args.mask_thresh, args.mask_nms_thresh,
            args.max_per_img, area_stats,
        )
        for p in final:
            results.append({
                "image_id": image_id,
                "category_id": int(p["label"]),
                "bbox": xyxy_to_xywh(p["box"]),
                "segmentation": mask_to_rle(p["mask"]),
                "score": float(p["score"]),
            })

    sp = os.path.join(args.output_dir, "test-results.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"\nSaved: {sp}, total {len(results)} preds")


def validate(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    val_json = os.path.join(args.ann_dir, "val.json")
    val_ds = CellDataset(
        img_dir=args.train_dir, ann_json_path=val_json, is_train=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    area_stats = None
    asp = os.path.join(args.output_dir, "area_stats.json")
    if args.use_area_filter and os.path.exists(asp):
        with open(asp, "r", encoding="utf-8") as f:
            area_stats = json.load(f)
    elif args.use_area_filter:
        area_stats = DEFAULT_AREA_STATS

    model = build_model(
        num_classes=args.num_classes, detector=args.detector,
        backbone=args.backbone, use_coco_pretrained=args.use_coco_pretrained,
        box_score_thresh=args.model_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        detections_per_img=args.max_per_img,
        anchor_preset=args.anchor_preset,
        drop_path_rate=args.drop_path_rate,
        rpn_nms_thresh=args.rpn_nms_thresh,
        min_size=args.crop_size, max_size=args.crop_size,
    )
    model = load_model_weights(model, args.checkpoint, device).to(device).eval()

    ap50, cm = evaluate_ap50(
        model, val_ds, val_loader, device,
        score_thresh=args.score_thresh, mask_thresh=args.mask_thresh,
        mask_nms_thresh=args.mask_nms_thresh, max_per_img=args.max_per_img,
        use_tta=args.use_tta, area_stats=area_stats, num_classes=4,
        cm_iou_thresh=0.5, compute_cm=True,
    )
    cm_names = ["background"] + [CATEGORY_NAMES[i] for i in [1, 2, 3, 4]]
    cm_path = os.path.join(args.output_dir, "confusion_matrix_validate.png")
    plot_confusion_matrix(
        cm, cm_names, cm_path, title=f"CM (validate, AP50={ap50:.4f})",
    )
    print(f"Validation AP50 = {ap50:.4f}, saved CM: {cm_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="train",
                   choices=["train", "inference", "validate"])

    p.add_argument("--train_dir", default="../data/train")
    p.add_argument("--test_dir", default="../data/test")
    p.add_argument("--ann_dir", default="../data/annotations")
    p.add_argument("--output_dir", default="./outputs_cascade")
    p.add_argument("--checkpoint", default="./outputs_cascade/best_ap50.pth")
    p.add_argument("--test_ids_json",
                   default="../data/test_image_name_to_ids.json")

    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument("--detector", default="cascade_mask_rcnn",
                   choices=["cascade_mask_rcnn", "mask_rcnn"])
    p.add_argument("--backbone", default="convnextv2_base",
                   choices=["convnextv2_base", "resnet50"])
    p.add_argument("--anchor_preset", default="youzhe",
                   choices=["youzhe", "default", "area_light", "area"])
    p.add_argument("--drop_path_rate", type=float, default=0.1)
    p.add_argument("--rpn_nms_thresh", type=float, default=0.45)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--val_interval", type=int, default=5)
    p.add_argument("--val_at_epoch1", action="store_true")
    p.add_argument("--compute_val_loss", action="store_true")
    p.add_argument("--train_compute_cm", action="store_true")

    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--score_thresh", type=float, default=0.05)
    p.add_argument("--mask_thresh", type=float, default=0.5)
    p.add_argument("--model_score_thresh", type=float, default=0.05)
    p.add_argument("--box_nms_thresh", type=float, default=0.5)
    p.add_argument("--mask_nms_thresh", type=float, default=0.5)
    p.add_argument("--max_per_img", type=int, default=300)

    p.add_argument("--crop_size", type=int, default=1024)
    p.add_argument("--use_random_crop", action="store_true")
    p.add_argument("--use_random_resize", action="store_true")
    p.add_argument("--use_elastic", action="store_true")
    p.add_argument("--use_copy_paste", action="store_true")
    p.add_argument("--copy_paste_prob", type=float, default=0.5)
    p.add_argument("--max_paste_objects", type=int, default=8)

    p.add_argument("--use_tta", action="store_true")
    p.add_argument("--use_area_filter", action="store_true")
    p.add_argument("--use_albu", action="store_true")
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--use_coco_pretrained", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "inference":
        inference(args)
    elif args.mode == "validate":
        validate(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
