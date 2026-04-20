import argparse
import contextlib
import copy
import io
import json
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import nms
from tqdm import tqdm


# ===================== Args =====================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_img_dir", default="/home/yunching/dl_vision/nycu-hw2-data/train")
    p.add_argument("--train_ann", default="/home/yunching/dl_vision/nycu-hw2-data/train.json")
    p.add_argument("--val_img_dir", default="/home/yunching/dl_vision/nycu-hw2-data/valid")
    p.add_argument("--val_ann", default="/home/yunching/dl_vision/nycu-hw2-data/valid.json")
    p.add_argument("--test_img_dir", default="/home/yunching/dl_vision/nycu-hw2-data/test")
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--pred_file", default="pred.json")
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_infer", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--img_size", type=int, default=512)

    # model
    p.add_argument("--num_queries", type=int, default=75)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--enc_layers", type=int, default=6)
    p.add_argument("--dec_layers", type=int, default=6)
    p.add_argument("--dim_feedforward", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n_points", type=int, default=4)

    # DN
    p.add_argument("--use_dn", action="store_true")
    p.add_argument("--dn_number", type=int, default=5)
    p.add_argument("--label_noise_ratio", type=float, default=0.2)
    p.add_argument("--box_noise_scale", type=float, default=0.4)
    p.add_argument("--dn_loss_coef", type=float, default=1.0)

    # augmentation
    p.add_argument("--mosaic_p", type=float, default=0.5, help="mosaic probability per sample")
    p.add_argument("--multi_scale", nargs="+", type=int, default=[448, 480, 512, 544, 576, 608])
    p.add_argument("--random_erase_p", type=float, default=0.15)

    # training
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_max_norm", type=float, default=0.1)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--ema_decay", type=float, default=0.9997)

    # loss / matcher
    p.add_argument("--cost_class", type=float, default=2.0)
    p.add_argument("--cost_bbox", type=float, default=5.0)
    p.add_argument("--cost_giou", type=float, default=2.0)
    p.add_argument("--loss_ce", type=float, default=1.0)
    p.add_argument("--loss_bbox", type=float, default=5.0)
    p.add_argument("--loss_giou", type=float, default=2.0)
    p.add_argument("--eos_coef", type=float, default=0.1)
    p.add_argument("--use_focal", action="store_true")
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--focal_prior", type=float, default=0.01)

    # eval / infer
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--val_score_thresh", type=float, default=0.05)
    p.add_argument("--score_thresh", type=float, default=0.3)
    p.add_argument("--nms_thresh", type=float, default=0.5)

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ===================== Utilities =====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_autocast(device, enabled=True):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    return contextlib.nullcontext()

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


# ===================== EMA =====================
class ModelEMA:
    def __init__(self, model, decay=0.9997):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
        for ema_b, model_b in zip(self.ema.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)


# ===================== Resize with Padding =====================
def resize_with_pad(img, target_size):
    w, h = img.size
    scale = target_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    padded = Image.new("RGB", (target_size, target_size), (128, 128, 128))
    padded.paste(img, (pad_left, pad_top))
    return padded, scale, pad_left, pad_top


# ===================== Mosaic =====================
def load_image_and_boxes(img_dir, img_info, anns):
    """Load image and return absolute bbox list [(x1,y1,x2,y2,label), ...]."""
    img = Image.open(os.path.join(img_dir, img_info["file_name"])).convert("RGB")
    w, h = img.size
    boxes = []
    for ann in anns:
        x, y, bw, bh = ann["bbox"]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        if x2 - x1 > 1 and y2 - y1 > 1:
            boxes.append((x1, y1, x2, y2, int(ann["category_id"])))
    return img, boxes


def make_mosaic(img_dir, img_infos, anns_list, target_size):
    """
    4-image mosaic. Place 4 images into a target_size × target_size canvas.
    Returns: PIL Image, list of (cx, cy, w, h, label) in normalized [0,1] coords.
    """
    s = target_size
    # Random center for the 4-image split
    cx = random.randint(int(s * 0.25), int(s * 0.75))
    cy = random.randint(int(s * 0.25), int(s * 0.75))

    canvas = Image.new("RGB", (s, s), (128, 128, 128))
    all_boxes = []

    # quadrant placement: (x_start, y_start, width, height)
    placements = [
        (0, 0, cx, cy),           # top-left
        (cx, 0, s - cx, cy),      # top-right
        (0, cy, cx, s - cy),      # bottom-left
        (cx, cy, s - cx, s - cy), # bottom-right
    ]

    for i, (px, py, pw, ph) in enumerate(placements):
        if pw < 2 or ph < 2:
            continue
        img, boxes = load_image_and_boxes(img_dir, img_infos[i], anns_list[i])
        orig_w, orig_h = img.size

        # Resize image to fit the quadrant
        scale_w = pw / orig_w
        scale_h = ph / orig_h
        scale = min(scale_w, scale_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        # Paste into canvas
        canvas.paste(img_resized, (px, py))

        # Transform boxes to canvas coordinates
        for (x1, y1, x2, y2, label) in boxes:
            nx1 = x1 * scale + px
            ny1 = y1 * scale + py
            nx2 = x2 * scale + px
            ny2 = y2 * scale + py
            # Clip to canvas
            nx1 = max(0, min(s, nx1))
            ny1 = max(0, min(s, ny1))
            nx2 = max(0, min(s, nx2))
            ny2 = max(0, min(s, ny2))
            bw = nx2 - nx1
            bh = ny2 - ny1
            if bw > 2 and bh > 2:
                # Convert to normalized cxcywh
                ncx = (nx1 + bw / 2.0) / s
                ncy = (ny1 + bh / 2.0) / s
                nw = bw / s
                nh = bh / s
                all_boxes.append((ncx, ncy, nw, nh, label))

    return canvas, all_boxes


# ===================== Dataset =====================
class CocoDigitDataset(Dataset):
    def __init__(self, img_dir, ann_file, img_size=512, is_train=True,
                 mosaic_p=0.5, random_erase_p=0.15):
        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)
        self.img_dir = img_dir
        self.img_size = img_size
        self.is_train = is_train
        self.mosaic_p = mosaic_p if is_train else 0.0
        self.images = sorted(coco["images"], key=lambda x: x["id"])
        self.img_id_to_anns = defaultdict(list)
        if "annotations" in coco:
            for ann in coco["annotations"]:
                self.img_id_to_anns[ann["image_id"]].append(ann)

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.25, 0.05) if is_train else None
        self.random_erase = transforms.RandomErasing(
            p=random_erase_p, scale=(0.02, 0.15), ratio=(0.3, 3.3)
        ) if is_train else None

    def set_img_size(self, size):
        """Called per-epoch for multi-scale training."""
        self.img_size = size

    def __len__(self):
        return len(self.images)

    def _load_single(self, idx):
        """Load a single image with scale jitter, pad, and return normalized boxes."""
        img_info = self.images[idx]
        img = Image.open(os.path.join(self.img_dir, img_info["file_name"])).convert("RGB")
        orig_w, orig_h = img.size
        scale_factor = 1.0

        if self.is_train:
            scale_factor = random.uniform(0.8, 1.2)
            aug_w = max(1, int(round(orig_w * scale_factor)))
            aug_h = max(1, int(round(orig_h * scale_factor)))
            img = img.resize((aug_w, aug_h), Image.BILINEAR)
            orig_w, orig_h = img.size

        img, scale, pad_left, pad_top = resize_with_pad(img, self.img_size)

        anns = self.img_id_to_anns[img_info["id"]]
        boxes, labels = [], []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            if self.is_train:
                x *= scale_factor; y *= scale_factor
                bw *= scale_factor; bh *= scale_factor
            x_pad = x * scale + pad_left
            y_pad = y * scale + pad_top
            bw_pad = bw * scale
            bh_pad = bh * scale
            cx = (x_pad + bw_pad / 2.0) / self.img_size
            cy = (y_pad + bh_pad / 2.0) / self.img_size
            nw = bw_pad / self.img_size
            nh = bh_pad / self.img_size
            cx = max(0.0, min(1.0, cx)); cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw)); nh = max(0.0, min(1.0, nh))
            if nw > 1e-3 and nh > 1e-3:
                boxes.append([cx, cy, nw, nh])
                labels.append(int(ann["category_id"]))

        return img, boxes, labels, img_info, scale, pad_left, pad_top

    def __getitem__(self, idx):
        img_info = self.images[idx]

        # ---- Mosaic ----
        if self.is_train and random.random() < self.mosaic_p:
            other_indices = random.sample(range(len(self.images)), 3)
            all_indices = [idx] + other_indices
            infos = [self.images[i] for i in all_indices]
            anns_list = [self.img_id_to_anns[info["id"]] for info in infos]
            img, box_list = make_mosaic(self.img_dir, infos, anns_list, self.img_size)

            if self.color_jitter is not None:
                img = self.color_jitter(img)

            img = self.normalize(img)
            if self.random_erase is not None:
                img = self.random_erase(img)

            boxes = [[b[0], b[1], b[2], b[3]] for b in box_list]
            labels = [b[4] for b in box_list]

            return img, {
                "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
                "image_id": int(img_info["id"]),
                "orig_size": torch.tensor([self.img_size, self.img_size], dtype=torch.long),
                "scale": torch.tensor(1.0, dtype=torch.float32),
                "pad": torch.tensor([0, 0], dtype=torch.float32),
            }

        # ---- Normal loading ----
        img, boxes, labels, img_info, scale, pad_left, pad_top = self._load_single(idx)

        if self.color_jitter is not None:
            img = self.color_jitter(img)

        img = self.normalize(img)
        if self.random_erase is not None:
            img = self.random_erase(img)

        orig_h, orig_w = self.images[idx].get("height", self.img_size), self.images[idx].get("width", self.img_size)
        return img, {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
            "image_id": int(img_info["id"]),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "pad": torch.tensor([pad_left, pad_top], dtype=torch.float32),
        }


class TestDataset(Dataset):
    def __init__(self, img_dir, img_size=512):
        self.img_dir = img_dir
        self.img_size = img_size
        self.filenames = sorted(os.listdir(img_dir))
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        orig_w, orig_h = img.size
        img, scale, pad_left, pad_top = resize_with_pad(img, self.img_size)
        img = self.normalize(img)
        img_id = int(os.path.splitext(fname)[0])
        return img, img_id, orig_h, orig_w, scale, pad_left, pad_top


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, dim=0), list(targets)

def collate_fn_test(batch):
    imgs, ids, hs, ws, scales, pls, pts = zip(*batch)
    return torch.stack(imgs, dim=0), list(ids), list(hs), list(ws), list(scales), list(pls), list(pts)


# ===================== Box Utils =====================
def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dim=-1)

def generalized_box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(2)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-7)
    lt_enc = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    area_enc = (rb_enc - lt_enc).clamp(min=0).prod(2)
    return iou - (area_enc - union) / (area_enc + 1e-7)

def convert_to_orig_coords(boxes_cxcywh, img_size, scale, pad_left, pad_top, orig_w, orig_h):
    cx = boxes_cxcywh[:, 0] * img_size
    cy = boxes_cxcywh[:, 1] * img_size
    bw = boxes_cxcywh[:, 2] * img_size
    bh = boxes_cxcywh[:, 3] * img_size
    cx = (cx - pad_left) / scale; cy = (cy - pad_top) / scale
    bw = bw / scale; bh = bh / scale
    x1 = (cx - bw / 2.0).clamp(min=0); y1 = (cy - bh / 2.0).clamp(min=0)
    x2 = (cx + bw / 2.0).clamp(max=orig_w); y2 = (cy + bh / 2.0).clamp(max=orig_h)
    return torch.stack([x1, y1, (x2 - x1).clamp(min=0), (y2 - y1).clamp(min=0)], dim=-1)


# ===================== Positional Encoding =====================
def build_2d_sincos_position_embedding(h, w, dim, device):
    if dim % 4 != 0:
        raise ValueError(f"hidden_dim must be divisible by 4, got {dim}")
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device), indexing="ij")
    grid_x = (grid_x + 0.5) / max(w, 1); grid_y = (grid_y + 0.5) / max(h, 1)
    omega = torch.arange(dim // 4, dtype=torch.float32, device=device)
    omega = 1.0 / (10000 ** (omega / max(dim // 4, 1)))
    out_x = grid_x.flatten()[:, None] * omega[None, :] * 2.0 * math.pi
    out_y = grid_y.flatten()[:, None] * omega[None, :] * 2.0 * math.pi
    return torch.cat([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=1)


# ===================== Multi-Scale Deformable Attention =====================
class MSDeformableAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=4, n_points=4):
        super().__init__()
        self.d_model = d_model; self.n_heads = n_heads
        self.n_levels = n_levels; self.n_points = n_points
        self.head_dim = d_model // n_heads
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points): grid_init[:, :, i, :] *= (i + 1)
        with torch.no_grad(): self.sampling_offsets.bias.copy_(grid_init.reshape(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query, reference_points, value, spatial_shapes):
        bsz, len_q, _ = query.shape; _, len_v, _ = value.shape
        value = self.value_proj(value).view(bsz, len_v, self.n_heads, self.head_dim)
        offsets = self.sampling_offsets(query).view(bsz, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attn_weights = self.attention_weights(query).view(bsz, len_q, self.n_heads, self.n_levels * self.n_points)
        attn_weights = F.softmax(attn_weights, dim=-1).view(bsz, len_q, self.n_heads, self.n_levels, self.n_points)
        ref = reference_points[:, :, None, :, None, :]
        spatial_tensor = torch.tensor([[w, h] for h, w in spatial_shapes], dtype=torch.float32, device=query.device)
        sampling_locations = ref + offsets / spatial_tensor[None, None, None, :, None, :]
        sampling_grids = 2.0 * sampling_locations - 1.0
        split_sizes = [h * w for h, w in spatial_shapes]
        value_list = value.split(split_sizes, dim=1)
        sampled_values = []
        for lid, (h, w) in enumerate(spatial_shapes):
            value_l = value_list[lid].permute(0, 2, 3, 1).reshape(bsz * self.n_heads, self.head_dim, h, w)
            grid_l = sampling_grids[:, :, :, lid, :, :].permute(0, 2, 1, 3, 4).reshape(bsz * self.n_heads, len_q, self.n_points, 2)
            sampled_values.append(F.grid_sample(value_l, grid_l, mode="bilinear", padding_mode="zeros", align_corners=False))
        sampled_values = torch.cat(sampled_values, dim=-1)
        attn_weights = attn_weights.view(bsz, len_q, self.n_heads, self.n_levels * self.n_points)
        attn_weights = attn_weights.permute(0, 2, 1, 3).reshape(bsz * self.n_heads, 1, len_q, self.n_levels * self.n_points)
        output = (sampled_values * attn_weights).sum(dim=-1)
        output = output.view(bsz, self.n_heads, self.head_dim, len_q).permute(0, 3, 1, 2).reshape(bsz, len_q, self.d_model)
        return self.output_proj(output)


# ===================== Encoder / Decoder =====================
class DeformableEncoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=4, n_points=4, d_ffn=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MSDeformableAttention(d_model, n_heads, n_levels, n_points)
        self.dropout1 = nn.Dropout(dropout); self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_ffn, d_model))
        self.dropout2 = nn.Dropout(dropout); self.norm2 = nn.LayerNorm(d_model)
    def forward(self, src, pos, reference_points, spatial_shapes):
        src2 = self.self_attn(src + pos, reference_points, src, spatial_shapes)
        src = self.norm1(src + self.dropout1(src2))
        src = self.norm2(src + self.dropout2(self.ffn(src)))
        return src

class DeformableDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=4, n_points=4, d_ffn=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout); self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformableAttention(d_model, n_heads, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout); self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_ffn, d_model))
        self.dropout3 = nn.Dropout(dropout); self.norm3 = nn.LayerNorm(d_model)
    def forward(self, tgt, query_pos, memory, reference_points, spatial_shapes, attn_mask=None):
        q = k = tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt, attn_mask=attn_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.cross_attn(tgt + query_pos, reference_points, memory, spatial_shapes)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt = self.norm3(tgt + self.dropout3(self.ffn(tgt)))
        return tgt


# ===================== DN helpers =====================
def make_denoising_queries(targets, num_classes, dn_number, label_noise_ratio, box_noise_scale, device):
    if dn_number <= 0: return None
    batch_size = len(targets)
    gt_counts = [int(t["labels"].numel()) for t in targets]
    max_gt = max(gt_counts) if gt_counts else 0
    if max_gt == 0: return None
    pad_size = max_gt * dn_number
    dn_labels = torch.zeros((batch_size, pad_size), dtype=torch.long, device=device)
    dn_boxes = torch.zeros((batch_size, pad_size, 4), dtype=torch.float32, device=device)
    dn_mask = torch.zeros((batch_size, pad_size), dtype=torch.bool, device=device)
    dn_positive_idx = []
    for b, target in enumerate(targets):
        labels = target["labels"].to(device); boxes = target["boxes"].to(device)
        num_gt = labels.numel(); pairs = []
        if num_gt == 0: dn_positive_idx.append(pairs); continue
        for rep in range(dn_number):
            start = rep * max_gt
            noisy_labels = labels.clone()
            if label_noise_ratio > 0:
                mask = torch.rand(num_gt, device=device) < label_noise_ratio
                noisy_labels = torch.where(mask, torch.randint(1, num_classes + 1, (num_gt,), device=device), noisy_labels)
            noisy_boxes = boxes.clone()
            if box_noise_scale > 0:
                sign = torch.randint(0, 2, noisy_boxes.shape, device=device).float() * 2.0 - 1.0
                mag = torch.rand_like(noisy_boxes)
                wh = boxes[:, [2, 3, 2, 3]].clamp(min=1e-3)
                noisy_boxes = (noisy_boxes + sign * mag * wh * box_noise_scale).clamp(0.0, 1.0)
                noisy_boxes[:, 2:] = noisy_boxes[:, 2:].clamp(min=1e-3)
            dn_labels[b, start:start + num_gt] = noisy_labels
            dn_boxes[b, start:start + num_gt] = noisy_boxes
            dn_mask[b, start:start + num_gt] = True
            pairs.extend([(start + i, i) for i in range(num_gt)])
        dn_positive_idx.append(pairs)
    attn_mask = torch.zeros((pad_size, pad_size), dtype=torch.bool, device=device)
    for rep in range(dn_number):
        s = rep * max_gt; e = s + max_gt
        attn_mask[s:e, :s] = True; attn_mask[s:e, e:pad_size] = True
    return dn_labels, dn_boxes, {"pad_size": pad_size, "max_gt": max_gt, "dn_positive_idx": dn_positive_idx, "dn_mask": dn_mask, "attn_mask": attn_mask}


# ===================== Deformable DETR =====================
class DeformableDETR(nn.Module):
    def __init__(self, num_classes=10, num_queries=75, hidden_dim=256, nheads=8,
                 enc_layers=6, dec_layers=6, dim_feedforward=1024, dropout=0.1,
                 n_points=4, n_levels=4, use_dn=False, dn_number=5,
                 label_noise_ratio=0.2, box_noise_scale=0.4,
                 use_focal=False, focal_prior=0.01):
        super().__init__()
        self.num_queries = num_queries; self.hidden_dim = hidden_dim
        self.num_classes = num_classes; self.n_levels = n_levels
        self.use_dn = use_dn; self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio; self.box_noise_scale = box_noise_scale
        self.use_focal = use_focal
        self.cls_out_dim = num_classes if use_focal else num_classes + 1

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2; self.layer3 = backbone.layer3; self.layer4 = backbone.layer4
        for module in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters(): p.requires_grad = False

        self.input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(512, hidden_dim, 1), nn.GroupNorm(32, hidden_dim)),
            nn.Sequential(nn.Conv2d(1024, hidden_dim, 1), nn.GroupNorm(32, hidden_dim)),
            nn.Sequential(nn.Conv2d(2048, hidden_dim, 1), nn.GroupNorm(32, hidden_dim)),
            nn.Sequential(nn.Conv2d(2048, hidden_dim, 3, stride=2, padding=1), nn.GroupNorm(32, hidden_dim)),
        ])
        self.level_embed = nn.Parameter(torch.randn(n_levels, hidden_dim))
        self.encoder_layers = nn.ModuleList([DeformableEncoderLayer(hidden_dim, nheads, n_levels, n_points, dim_feedforward, dropout) for _ in range(enc_layers)])
        self.decoder_layers = nn.ModuleList([DeformableDecoderLayer(hidden_dim, nheads, n_levels, n_points, dim_feedforward, dropout) for _ in range(dec_layers)])
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        self.reference_points_head = nn.Linear(hidden_dim, 4)
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim)
        self.box_enc = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim))
        self.dn_query_pos = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim))
        self.class_heads = nn.ModuleList([nn.Linear(hidden_dim, self.cls_out_dim) for _ in range(dec_layers)])
        self.bbox_heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 4)) for _ in range(dec_layers)])
        if use_focal:
            bias_value = -math.log((1 - focal_prior) / focal_prior)
            for head in self.class_heads: nn.init.constant_(head.bias, bias_value)

    def _get_encoder_reference_points(self, spatial_shapes, device):
        refs = []
        for h, w in spatial_shapes:
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, h - 0.5, h, device=device) / h, torch.linspace(0.5, w - 0.5, w, device=device) / w, indexing="ij")
            refs.append(torch.stack([ref_x.reshape(-1), ref_y.reshape(-1)], dim=-1))
        return torch.cat(refs, dim=0)[:, None, :].repeat(1, len(spatial_shapes), 1)

    def _build_dn_inputs(self, targets, device):
        if not self.training or not self.use_dn or targets is None: return None, None, None, None
        out = make_denoising_queries(targets, self.num_classes, self.dn_number, self.label_noise_ratio, self.box_noise_scale, device)
        if out is None: return None, None, None, None
        dn_labels, dn_boxes, dn_meta = out
        return self.label_enc(dn_labels) + self.box_enc(dn_boxes), self.dn_query_pos(dn_boxes), dn_boxes, dn_meta

    def forward(self, x, targets=None):
        bsz = x.size(0); device = x.device
        c2 = self.layer1(x); c3 = self.layer2(c2); c4 = self.layer3(c3); c5 = self.layer4(c4)
        srcs, poss, spatial_shapes = [], [], []
        for lid, feat in enumerate([c3, c4, c5, c5]):
            src = self.input_proj[lid](feat); _, _, h, w = src.shape; spatial_shapes.append((h, w))
            pos = build_2d_sincos_position_embedding(h, w, self.hidden_dim, device).unsqueeze(0).expand(bsz, -1, -1) + self.level_embed[lid].view(1, 1, -1)
            srcs.append(src.flatten(2).permute(0, 2, 1)); poss.append(pos)
        src_flat = torch.cat(srcs, dim=1); pos_flat = torch.cat(poss, dim=1)
        enc_ref = self._get_encoder_reference_points(spatial_shapes, device).unsqueeze(0).expand(bsz, -1, -1, -1)
        memory = src_flat
        for layer in self.encoder_layers: memory = layer(memory, pos_flat, enc_ref, spatial_shapes)

        qe = self.query_embed.weight; qp, qc = qe.split(self.hidden_dim, dim=-1)
        query_pos = qp.unsqueeze(0).expand(bsz, -1, -1); tgt = qc.unsqueeze(0).expand(bsz, -1, -1)
        reference_points = self.reference_points_head(query_pos).sigmoid()

        dn_tgt, dn_qpos, dn_ref, dn_meta = self._build_dn_inputs(targets, device)
        total_attn_mask = None
        if dn_tgt is not None:
            tgt = torch.cat([dn_tgt, tgt], dim=1); query_pos = torch.cat([dn_qpos, query_pos], dim=1)
            reference_points = torch.cat([dn_ref, reference_points], dim=1)
            total_len = tgt.size(1); pad_size = dn_meta["pad_size"]
            total_attn_mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)
            total_attn_mask[:pad_size, :pad_size] = dn_meta["attn_mask"]; total_attn_mask[pad_size:, :pad_size] = True

        outputs_classes, outputs_coords = [], []; output = tgt
        for lid, layer in enumerate(self.decoder_layers):
            output = layer(output, query_pos, memory, reference_points[:, :, None, :2].repeat(1, 1, self.n_levels, 1), spatial_shapes, attn_mask=total_attn_mask)
            output_norm = self.decoder_norm(output)
            pred_boxes = (self.bbox_heads[lid](output_norm) + inverse_sigmoid(reference_points)).sigmoid()
            outputs_classes.append(self.class_heads[lid](output_norm)); outputs_coords.append(pred_boxes)
            reference_points = pred_boxes.detach()

        out = {"pred_logits": outputs_classes[-1], "pred_boxes": outputs_coords[-1],
               "aux_outputs": [{"pred_logits": outputs_classes[i], "pred_boxes": outputs_coords[i]} for i in range(len(outputs_classes) - 1)]}
        if dn_meta is not None: out["dn_meta"] = dn_meta
        return out

    def train(self, mode=True):
        super().train(mode)
        for module in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d): m.eval()
        return self


# ===================== Matcher =====================
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.cost_class = cost_class; self.cost_bbox = cost_bbox; self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha; self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs, targets, use_focal=False):
        bsz = outputs["pred_logits"].shape[0]; indices = []
        for b in range(bsz):
            tgt_labels = targets[b]["labels"]; tgt_boxes = targets[b]["boxes"]
            if tgt_labels.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))); continue
            pred_box = outputs["pred_boxes"][b]
            if use_focal:
                out_prob = outputs["pred_logits"][b].sigmoid()
                neg_cost = self.focal_alpha * (out_prob ** self.focal_gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost = (1 - self.focal_alpha) * ((1 - out_prob) ** self.focal_gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost[:, tgt_labels - 1] - neg_cost[:, tgt_labels - 1]
            else:
                cost_class = -outputs["pred_logits"][b].softmax(-1)[:, tgt_labels]
            cost_bbox = torch.cdist(pred_box, tgt_boxes, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(pred_box), box_cxcywh_to_xyxy(tgt_boxes))
            cost = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            r, c = linear_sum_assignment(cost.cpu().numpy())
            indices.append((torch.as_tensor(r, dtype=torch.long), torch.as_tensor(c, dtype=torch.long)))
        return indices


# ===================== Loss =====================
def sigmoid_focal_loss(logits, targets_onehot, alpha=0.25, gamma=2.0, num_boxes=1):
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets_onehot, reduction="none")
    p_t = prob * targets_onehot + (1 - prob) * (1 - targets_onehot)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets_onehot + (1 - alpha) * (1 - targets_onehot)
        loss = alpha_t * loss
    return loss.mean(1).sum() / max(num_boxes, 1)


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, w_ce=1.0, w_bbox=5.0, w_giou=2.0, eos_coef=0.1,
                 dn_loss_coef=1.0, use_focal=False, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes; self.matcher = matcher
        self.w_ce = w_ce; self.w_bbox = w_bbox; self.w_giou = w_giou
        self.dn_loss_coef = dn_loss_coef; self.use_focal = use_focal
        self.focal_alpha = focal_alpha; self.focal_gamma = focal_gamma
        ew = torch.ones(num_classes + 1); ew[0] = eos_coef
        self.register_buffer("empty_weight", ew)

    def _compute_loss(self, outputs, targets, indices):
        device = outputs["pred_logits"].device
        num_boxes = max(sum(len(s) for s, _ in indices), 1)
        if self.use_focal:
            B, Q, C = outputs["pred_logits"].shape
            tgt_oh = torch.zeros((B, Q, C), dtype=outputs["pred_logits"].dtype, device=device)
            for b, (si, ti) in enumerate(indices):
                if len(si) > 0: tgt_oh[b, si, targets[b]["labels"][ti].to(device) - 1] = 1.0
            loss_ce = sigmoid_focal_loss(outputs["pred_logits"], tgt_oh, self.focal_alpha, self.focal_gamma, num_boxes)
        else:
            tc = torch.zeros(outputs["pred_logits"].shape[:2], dtype=torch.long, device=device)
            for b, (si, ti) in enumerate(indices):
                if len(si) > 0: tc[b, si] = targets[b]["labels"][ti]
            loss_ce = F.cross_entropy(outputs["pred_logits"].permute(0, 2, 1), tc, weight=self.empty_weight)

        sb, tb = [], []
        for b, (si, ti) in enumerate(indices):
            if len(si) > 0: sb.append(outputs["pred_boxes"][b][si]); tb.append(targets[b]["boxes"][ti])
        if sb:
            sb = torch.cat(sb); tb = torch.cat(tb)
            loss_bbox = F.l1_loss(sb, tb, reduction="sum") / num_boxes
            loss_giou = (1.0 - generalized_box_iou(box_cxcywh_to_xyxy(sb), box_cxcywh_to_xyxy(tb)).diag()).sum() / num_boxes
        else:
            loss_bbox = loss_giou = torch.tensor(0.0, device=device)
        total = self.w_ce * loss_ce + self.w_bbox * loss_bbox + self.w_giou * loss_giou
        return total, float(loss_ce.item()), float(loss_bbox.item()), float(loss_giou.item())

    def _compute_dn_loss(self, outputs, targets, dn_meta):
        device = outputs["pred_logits"].device; pad_size = int(dn_meta["pad_size"])
        if pad_size <= 0: return torch.tensor(0.0, device=device), 0.0
        pl = outputs["pred_logits"][:, :pad_size]; pb = outputs["pred_boxes"][:, :pad_size]
        sl, tl, sbox, tbox, sid_list = [], [], [], [], []
        for b, pairs in enumerate(dn_meta["dn_positive_idx"]):
            if not pairs: sid_list.append(torch.empty(0, dtype=torch.long, device=device)); continue
            si = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
            ti = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
            sid_list.append(si); sl.append(pl[b, si]); tl.append(targets[b]["labels"][ti].to(device))
            sbox.append(pb[b, si]); tbox.append(targets[b]["boxes"][ti].to(device))
        if not sl: return torch.tensor(0.0, device=device), 0.0
        nb = max(sum(s.numel() for s in sid_list), 1)
        if self.use_focal:
            B, _, C = pl.shape; tgt_oh = torch.zeros_like(pl)
            for b, si in enumerate(sid_list):
                if si.numel() == 0: continue
                pairs = dn_meta["dn_positive_idx"][b]
                ti = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
                tgt_oh[b, si, targets[b]["labels"][ti].to(device) - 1] = 1.0
            lce = sigmoid_focal_loss(pl, tgt_oh, self.focal_alpha, self.focal_gamma, nb)
        else:
            sl_cat = torch.cat(sl); tl_cat = torch.cat(tl)
            lce = F.cross_entropy(sl_cat, tl_cat, weight=self.empty_weight)
        sbox = torch.cat(sbox); tbox = torch.cat(tbox)
        lb = F.l1_loss(sbox, tbox, reduction="sum") / nb
        lg = (1.0 - generalized_box_iou(box_cxcywh_to_xyxy(sbox), box_cxcywh_to_xyxy(tbox)).diag()).sum() / nb
        total = (self.w_ce * lce + self.w_bbox * lb + self.w_giou * lg) * self.dn_loss_coef
        return total, float(total.item())

    def _split(self, outputs):
        dm = outputs.get("dn_meta")
        if dm is None: return outputs, None
        ps = int(dm["pad_size"])
        mo = {"pred_logits": outputs["pred_logits"][:, ps:], "pred_boxes": outputs["pred_boxes"][:, ps:]}
        if "aux_outputs" in outputs:
            mo["aux_outputs"] = [{"pred_logits": a["pred_logits"][:, ps:], "pred_boxes": a["pred_boxes"][:, ps:]} for a in outputs["aux_outputs"]]
        return mo, dm

    def forward(self, outputs, targets):
        mo, dm = self._split(outputs)
        indices = self.matcher(mo, targets, use_focal=self.use_focal)
        lm, ce, bb, gi = self._compute_loss(mo, targets, indices)
        la = torch.tensor(0.0, device=lm.device)
        if "aux_outputs" in mo:
            for aux in mo["aux_outputs"]:
                ai = self.matcher(aux, targets, use_focal=self.use_focal)
                al, _, _, _ = self._compute_loss(aux, targets, ai); la = la + al
        ld = torch.tensor(0.0, device=lm.device); dv = 0.0
        if dm is not None: ld, dv = self._compute_dn_loss(outputs, targets, dm)
        total = lm + la + ld
        return total, {"loss": float(total.item()), "loss_main": float(lm.item()), "loss_aux": float(la.item()), "loss_ce": ce, "loss_bbox": bb, "loss_giou": gi, "loss_dn": dv}, indices


# ===================== Postprocess =====================
def postprocess_single_image_predictions(logits, boxes, img_size, scale, pad_left, pad_top, orig_w, orig_h, score_thresh, nms_thresh, use_focal=False):
    if use_focal:
        probs = logits.sigmoid(); scores, ci = probs.max(dim=-1); cls_ids = ci + 1
    else:
        probs = logits.softmax(-1); scores, ci = probs[:, 1:].max(dim=-1); cls_ids = ci + 1
    keep = scores > score_thresh; scores = scores[keep]; cls_ids = cls_ids[keep]; boxes = boxes[keep]
    if scores.numel() == 0: return []
    bx = convert_to_orig_coords(boxes.cpu(), img_size, scale, pad_left, pad_top, orig_w, orig_h)
    bxy = bx.clone(); bxy[:, 2] = bxy[:, 0] + bxy[:, 2]; bxy[:, 3] = bxy[:, 1] + bxy[:, 3]
    ki = []
    for cls in cls_ids.unique():
        cm = (cls_ids == cls).nonzero(as_tuple=True)[0]
        ki.append(cm[nms(bxy[cm].float(), scores[cm].float(), nms_thresh)])
    if not ki: return []
    ki = torch.cat(ki)
    return [{"bbox": bx[i].tolist(), "score": float(scores[i]), "category_id": int(cls_ids[i])} for i in ki.tolist()]


# ===================== Evaluation =====================
@torch.no_grad()
def evaluate(model, loader, criterion, device, img_size, output_dir, epoch, num_classes, val_score_thresh, nms_thresh, use_focal):
    model.eval(); tl = defaultdict(float); nb = 0
    ap, ag = [], {"images": [], "annotations": [], "categories": [{"id": i} for i in range(1, num_classes + 1)]}
    gid = 0; apc, agc = [], []; use_amp = device.type == "cuda"
    for imgs, targets in tqdm(loader, desc="Eval", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with get_autocast(device, enabled=use_amp):
            outputs = model(imgs); loss, ld, indices = criterion(outputs, targets)
        ml = outputs["pred_logits"]
        if "dn_meta" in outputs: ml = ml[:, int(outputs["dn_meta"]["pad_size"]):]
        for k, v in ld.items(): tl[k] += float(v)
        nb += 1
        if use_focal: pc = ml.sigmoid().argmax(dim=-1) + 1
        else: pc = ml.argmax(dim=-1)
        for b, (si, ti) in enumerate(indices):
            if len(si) > 0: apc.extend(pc[b][si].cpu().tolist()); agc.extend(targets[b]["labels"][ti].cpu().tolist())
        pb = outputs["pred_boxes"]
        if "dn_meta" in outputs: pb = pb[:, int(outputs["dn_meta"]["pad_size"]):]
        for b in range(imgs.size(0)):
            iid = int(targets[b]["image_id"]); oh, ow = targets[b]["orig_size"].tolist()
            sc = float(targets[b]["scale"]); pl, pt = targets[b]["pad"].tolist()
            ag["images"].append({"id": iid, "width": int(ow), "height": int(oh)})
            for j in range(len(targets[b]["labels"])):
                ba = convert_to_orig_coords(targets[b]["boxes"][j:j+1].cpu(), img_size, sc, pl, pt, ow, oh)[0].tolist()
                ag["annotations"].append({"id": gid, "image_id": iid, "category_id": int(targets[b]["labels"][j]), "bbox": ba, "area": float(ba[2]*ba[3]), "iscrowd": 0}); gid += 1
            preds = postprocess_single_image_predictions(ml[b].cpu(), pb[b].cpu(), img_size, sc, pl, pt, ow, oh, val_score_thresh, nms_thresh, use_focal)
            for p in preds: ap.append({"image_id": iid, "category_id": p["category_id"], "bbox": p["bbox"], "score": p["score"]})
    al = {k: v / max(nb, 1) for k, v in tl.items()}
    mAP = 0.0
    try:
        from pycocotools.coco import COCO; from pycocotools.cocoeval import COCOeval
        cg = COCO(); cg.dataset = ag; cg.createIndex()
        if ap:
            cd = cg.loadRes(ap); ce = COCOeval(cg, cd, "bbox")
            with contextlib.redirect_stdout(io.StringIO()): ce.evaluate(); ce.accumulate()
            ce.summarize(); mAP = float(ce.stats[0])
    except Exception as e: print(f"mAP eval failed: {e}")
    if apc:
        plot_confusion_matrix(apc, agc, output_dir, epoch, num_classes)
        ma = sum(int(p == g) for p, g in zip(apc, agc)) / len(apc)
    else: ma = 0.0
    return al, ma, mAP


# ===================== Plotting =====================
def plot_confusion_matrix(preds, gts, output_dir, epoch, num_classes):
    try:
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
        labels = list(range(1, num_classes + 1)); cm = confusion_matrix(gts, preds, labels=labels)
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in labels]).plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Confusion Matrix (epoch {epoch + 1})"); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_ep{epoch + 1}.png"), dpi=150); plt.close(fig)
    except ImportError: pass

def plot_curves(history, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(history["train_loss"], label="Train Loss"); axes[0, 0].set_title("Train Loss"); axes[0, 0].grid(True); axes[0, 0].legend()
    ep = [i for i, v in enumerate(history["val_loss"]) if v is not None]; vl = [v for v in history["val_loss"] if v is not None]
    if vl: axes[0, 1].plot(ep, vl, label="Val Loss", marker="o", markersize=3)
    axes[0, 1].set_title("Val Loss"); axes[0, 1].grid(True); axes[0, 1].legend()
    ea = [i for i, v in enumerate(history["val_acc"]) if v is not None]; va = [v for v in history["val_acc"] if v is not None]
    if va: axes[1, 0].plot(ea, va, label="Val Acc", marker="o", markersize=3)
    axes[1, 0].set_title("Val Matched Accuracy"); axes[1, 0].grid(True); axes[1, 0].legend()
    em = [i for i, v in enumerate(history["val_mAP"]) if v is not None]; vm = [v for v in history["val_mAP"] if v is not None]
    if vm: axes[1, 1].plot(em, vm, label="Val mAP", marker="o", markersize=3)
    axes[1, 1].set_title("Val mAP @[.5:.95]"); axes[1, 1].grid(True); axes[1, 1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "curves.png"), dpi=150); plt.close(fig)

def build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.05):
    def lr_lambda(epoch):
        if epoch < warmup_epochs: return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ===================== Inference =====================
@torch.no_grad()
def inference(model, test_dir, img_size, device, score_thresh, nms_thresh, output_path, use_focal):
    model.eval(); ds = TestDataset(test_dir, img_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn_test, pin_memory=(device.type == "cuda"))
    results = []; use_amp = device.type == "cuda"
    for imgs, img_ids, hs, ws, scales, pls, pts in tqdm(loader, desc="Inference"):
        imgs = imgs.to(device, non_blocking=True)
        with get_autocast(device, enabled=use_amp): outputs = model(imgs)
        preds = postprocess_single_image_predictions(outputs["pred_logits"][0].cpu(), outputs["pred_boxes"][0].cpu(), img_size, scales[0], pls[0], pts[0], ws[0], hs[0], score_thresh, nms_thresh, use_focal)
        for p in preds: results.append({"image_id": int(img_ids[0]), "bbox": [round(v, 2) for v in p["bbox"]], "score": round(p["score"], 6), "category_id": p["category_id"]})
    with open(output_path, "w") as f: json.dump(results, f)
    print(f"Saved {len(results)} predictions to {output_path}")


# ===================== Main =====================
def main():
    args = parse_args(); os.makedirs(args.output_dir, exist_ok=True); set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | img_size={args.img_size} | use_dn={args.use_dn} | use_focal={args.use_focal}")
    print(f"Mosaic p={args.mosaic_p} | Multi-scale={args.multi_scale} | RandomErase p={args.random_erase_p}")

    model = DeformableDETR(
        num_classes=args.num_classes, num_queries=args.num_queries, hidden_dim=args.hidden_dim,
        nheads=args.nheads, enc_layers=args.enc_layers, dec_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout, n_points=args.n_points,
        use_dn=args.use_dn, dn_number=args.dn_number,
        label_noise_ratio=args.label_noise_ratio, box_noise_scale=args.box_noise_scale,
        use_focal=args.use_focal, focal_prior=args.focal_prior,
    ).to(device)
    ema = ModelEMA(model, decay=args.ema_decay)

    start_epoch = 0; best_mAP = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_mAP": []}; ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "ema" in ckpt: ema.ema.load_state_dict(ckpt["ema"])
        print(f"Loaded: {args.resume}")
        if "epoch" in ckpt: start_epoch = int(ckpt["epoch"]) + 1
        if "best_mAP" in ckpt: best_mAP = float(ckpt["best_mAP"])
        if "history" in ckpt: history = ckpt["history"]

    if args.do_train:
        train_ds = CocoDigitDataset(args.train_img_dir, args.train_ann, args.img_size, is_train=True, mosaic_p=args.mosaic_p, random_erase_p=args.random_erase_p)
        val_ds = CocoDigitDataset(args.val_img_dir, args.val_ann, args.img_size, is_train=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=(device.type == "cuda"))

        bp = ("layer1", "layer2", "layer3", "layer4")
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if not n.startswith(bp) and p.requires_grad], "lr": args.lr},
            {"params": [p for n, p in model.named_parameters() if n.startswith(bp) and p.requires_grad], "lr": args.lr_backbone},
        ]
        optimizer = torch.optim.AdamW(param_dicts, weight_decay=args.weight_decay)
        scheduler = build_warmup_cosine_scheduler(optimizer, args.warmup_epochs, args.epochs, args.min_lr_ratio)
        use_amp = device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        if ckpt and "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt and "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])

        matcher = HungarianMatcher(args.cost_class, args.cost_bbox, args.cost_giou, args.focal_alpha, args.focal_gamma)
        criterion = SetCriterion(args.num_classes, matcher, args.loss_ce, args.loss_bbox, args.loss_giou, args.eos_coef,
                                 dn_loss_coef=args.dn_loss_coef, use_focal=args.use_focal, focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma).to(device)

        for epoch in range(start_epoch, args.epochs):
            # ---- Multi-scale: pick random size per epoch ----
            if args.multi_scale and len(args.multi_scale) > 1:
                ms = random.choice(args.multi_scale)
                train_ds.set_img_size(ms)
                print(f"  Multi-scale: {ms}x{ms}")

            model.train(); el = 0.0; nb = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for imgs, targets in pbar:
                imgs = imgs.to(device, non_blocking=True)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                optimizer.zero_grad(set_to_none=True)
                with get_autocast(device, enabled=use_amp):
                    outputs = model(imgs, targets if args.use_dn else None)
                    loss, ld, _ = criterion(outputs, targets)
                scaler.scale(loss).backward()
                if args.clip_max_norm > 0: scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                scaler.step(optimizer); scaler.update(); ema.update(model)
                el += ld["loss"]; nb += 1
                pbar.set_postfix(loss=f"{ld['loss']:.4f}", ce=f"{ld['loss_ce']:.3f}", bbox=f"{ld['loss_bbox']:.3f}", giou=f"{ld['loss_giou']:.3f}", dn=f"{ld['loss_dn']:.3f}")

            scheduler.step(); atl = el / max(nb, 1); history["train_loss"].append(atl)

            # ---- Reset val dataset to default img_size ----
            train_ds.set_img_size(args.img_size)

            if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
                vl, va, vm = evaluate(ema.ema, val_loader, criterion, device, args.img_size, args.output_dir, epoch, args.num_classes, args.val_score_thresh, args.nms_thresh, args.use_focal)
                history["val_loss"].append(vl.get("loss", 0.0)); history["val_acc"].append(va); history["val_mAP"].append(vm)
                print(f"Epoch {epoch + 1} | Train: {atl:.4f} | Val: {vl.get('loss', 0.0):.4f} | Acc: {va:.4f} | mAP(EMA): {vm:.4f}")
                if vm > best_mAP:
                    best_mAP = vm
                    torch.save({"epoch": epoch, "model": model.state_dict(), "ema": ema.ema.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "best_mAP": best_mAP, "history": history}, os.path.join(args.output_dir, "best.pth"))
            else:
                history["val_loss"].append(None); history["val_acc"].append(None); history["val_mAP"].append(None)
                print(f"Epoch {epoch + 1} | Train: {atl:.4f}")

            torch.save({"epoch": epoch, "model": model.state_dict(), "ema": ema.ema.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "best_mAP": best_mAP, "history": history}, os.path.join(args.output_dir, "latest.pth"))
            if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1: plot_curves(history, args.output_dir)

        plot_curves(history, args.output_dir); print(f"Best mAP: {best_mAP:.4f}")

    if args.do_infer:
        if not args.do_train and args.resume is None: raise ValueError("--do_infer requires --resume")
        inference(ema.ema, args.test_img_dir, args.img_size, device, args.score_thresh, args.nms_thresh, os.path.join(args.output_dir, args.pred_file), args.use_focal)

if __name__ == "__main__":
    main()