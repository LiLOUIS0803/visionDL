import argparse
import csv
import os
import random
import zipfile
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Utils
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_num_blocks(s):
    if isinstance(s, (tuple, list)):
        return tuple(s)

    parts = [int(x.strip()) for x in str(s).split(",")]

    if len(parts) != 4:
        raise ValueError("--num_blocks must be like '4,6,6,8' for full PromptIR")

    return tuple(parts)


def load_rgb(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0

    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def tensor_to_uint8_chw(x):
    x = x.detach().float().cpu().clamp(0, 1)
    x = (x * 255.0).round().byte().numpy()
    return x


def calc_psnr(pred, target, eps=1e-8):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)

    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def chw_to_hwc(x):
    return x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def append_log_csv(log_path, row):
    log_path = Path(log_path)
    write_header = not log_path.exists()

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def flexible_load_checkpoint(model, ckpt_path, device, use_ema=False):
    ckpt = torch.load(ckpt_path, map_location=device)

    if (
        use_ema
        and isinstance(ckpt, dict)
        and "ema_model" in ckpt
        and ckpt["ema_model"] is not None
    ):
        print("[Resume] Loading EMA weights.")
        state_dict = ckpt["ema_model"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        print("[Resume] Loading normal model weights.")
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model_dict = model.state_dict()
    matched = {}
    skipped = []

    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            matched[k] = v
        else:
            skipped.append(k)

    model_dict.update(matched)
    model.load_state_dict(model_dict, strict=True)

    print(f"[Resume] Loaded matched keys: {len(matched)}")
    print(f"[Resume] Skipped keys: {len(skipped)}")

    if len(skipped) > 0:
        print("[Resume] Example skipped keys:")
        for k in skipped[:10]:
            print(f"  - {k}")


# ============================================================
# EMA
# ============================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = deepcopy(model)
        self.ema_model.eval()

        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()

        for k in ema_state.keys():
            if not torch.is_floating_point(ema_state[k]):
                ema_state[k].copy_(model_state[k])
            else:
                ema_state[k].mul_(self.decay).add_(
                    model_state[k].detach(),
                    alpha=1.0 - self.decay,
                )

    def state_dict(self):
        return self.ema_model.state_dict()


# ============================================================
# Dataset
# ============================================================
class RestorationDataset(Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        val_ratio=0.1,
        patch_size=128,
        seed=42,
        residual_aug_prob=0.0,
        residual_pool_size=300,
        repeat_factor=1,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.patch_size = patch_size
        self.residual_aug_prob = residual_aug_prob
        self.residual_pool_size = residual_pool_size
        self.repeat_factor = max(1, int(repeat_factor))

        degraded_dir, clean_dir = self._find_train_dirs(self.data_root)

        print(f"[Dataset:{split}] degraded_dir = {degraded_dir}")
        print(f"[Dataset:{split}] clean_dir    = {clean_dir}")

        all_files = sorted(list(degraded_dir.glob("*.png")))

        rain_files = [p for p in all_files if p.name.lower().startswith("rain")]
        snow_files = [p for p in all_files if p.name.lower().startswith("snow")]

        print(f"[Dataset:{split}] total degraded png = {len(all_files)}")
        print(f"[Dataset:{split}] rain files = {len(rain_files)}")
        print(f"[Dataset:{split}] snow files = {len(snow_files)}")

        if len(all_files) == 0:
            raise RuntimeError(f"No png images found in: {degraded_dir}")

        if len(rain_files) == 0 and len(snow_files) == 0:
            raise RuntimeError(
                f"Found png files, but none starts with rain/snow.\n"
                f"Example files: {[p.name for p in all_files[:10]]}"
            )

        rng = random.Random(seed)
        rng.shuffle(rain_files)
        rng.shuffle(snow_files)

        self.clean_dir = clean_dir

        def split_files(files):
            val_n = int(len(files) * val_ratio)
            val_n = max(1, val_n) if len(files) > 0 else 0

            if split == "val":
                return files[:val_n]

            return files[val_n:]

        self.degraded_files = split_files(rain_files) + split_files(snow_files)

        if len(self.degraded_files) == 0:
            raise RuntimeError(
                f"Dataset split is empty. split={split}, val_ratio={val_ratio}"
            )

        self.residual_pools = {0: [], 1: []}

        if self.split == "train" and self.residual_aug_prob > 0:
            self._build_residual_pools(rain_files, snow_files)

    def _find_train_dirs(self, root):
        candidates = [
            root,
            root / "train",
            root / "Train",
        ]

        for base in candidates:
            degraded_dir = base / "degraded"
            clean_dir = base / "clean"

            if degraded_dir.exists() and clean_dir.exists():
                return degraded_dir, clean_dir

        degraded_candidates = list(root.rglob("degraded"))
        clean_candidates = list(root.rglob("clean"))

        if degraded_candidates and clean_candidates:
            return degraded_candidates[0], clean_candidates[0]

        raise FileNotFoundError(
            f"Cannot find degraded/clean folders under: {root}\n"
            f"Expected:\n"
            f"{root}/train/degraded\n"
            f"{root}/train/clean"
        )

    def __len__(self):
        if self.split == "train":
            return len(self.degraded_files) * self.repeat_factor
        return len(self.degraded_files)

    def _get_clean_path(self, degraded_path):
        name = degraded_path.name
        lower = name.lower()

        if lower.startswith("rain"):
            idx = (
                name.replace("rain-", "")
                .replace("rain_", "")
                .replace(".png", "")
            )
            possible_names = [
                f"rain_clean-{idx}.png",
                f"rain_clean_{idx}.png",
                f"rain-clean-{idx}.png",
            ]
        elif lower.startswith("snow"):
            idx = (
                name.replace("snow-", "")
                .replace("snow_", "")
                .replace(".png", "")
            )
            possible_names = [
                f"snow_clean-{idx}.png",
                f"snow_clean_{idx}.png",
                f"snow-clean-{idx}.png",
            ]
        else:
            raise ValueError(f"Unknown degraded filename: {name}")

        for clean_name in possible_names:
            clean_path = self.clean_dir / clean_name
            if clean_path.exists():
                return clean_path

        raise FileNotFoundError(
            f"Cannot find clean image for degraded image: {name}\n"
            f"Tried: {possible_names}\n"
            f"Clean dir: {self.clean_dir}"
        )

    def _get_degradation_label(self, degraded_path):
        name = degraded_path.name.lower()

        if name.startswith("rain"):
            return 0

        if name.startswith("snow"):
            return 1

        raise ValueError(f"Unknown degradation type: {name}")

    def _build_residual_pools(self, rain_files, snow_files):
        print("[Residual Aug] Building residual pools...")

        pool_config = [
            (0, rain_files, "rain"),
            (1, snow_files, "snow"),
        ]

        for label, files, name in pool_config:
            files = files[: self.residual_pool_size]

            for degraded_path in files:
                clean_path = self._get_clean_path(degraded_path)

                degraded = load_rgb(degraded_path)
                clean = load_rgb(clean_path)

                residual = degraded - clean

                _, h, w = residual.shape
                ps = self.patch_size

                if h >= ps and w >= ps:
                    top = random.randint(0, h - ps)
                    left = random.randint(0, w - ps)
                    residual = residual[:, top:top + ps, left:left + ps]
                else:
                    residual = F.interpolate(
                        residual.unsqueeze(0),
                        size=(ps, ps),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

                self.residual_pools[label].append(residual)

            print(
                f"[Residual Aug] {name} residual patches: "
                f"{len(self.residual_pools[label])}"
            )

    def _random_crop_pair(self, x, y):
        _, h, w = x.shape
        ps = self.patch_size

        if h < ps or w < ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)

            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            y = F.pad(y, (0, pad_w, 0, pad_h), mode="reflect")

            _, h, w = x.shape

        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)

        x = x[:, top:top + ps, left:left + ps]
        y = y[:, top:top + ps, left:left + ps]

        return x, y

    def _augment_pair(self, x, y):
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])

        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[1])

        if random.random() < 0.5:
            k = random.randint(1, 3)
            x = torch.rot90(x, k, dims=[1, 2])
            y = torch.rot90(y, k, dims=[1, 2])

        return x, y

    def _apply_same_prompt_residual_aug(self, degraded, clean, label):
        if self.split != "train":
            return degraded

        if random.random() > self.residual_aug_prob:
            return degraded

        pool = self.residual_pools.get(label, [])

        if len(pool) == 0:
            return degraded

        residual = random.choice(pool)

        _, h, w = clean.shape
        _, rh, rw = residual.shape

        if rh != h or rw != w:
            residual = F.interpolate(
                residual.unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        alpha = random.uniform(0.5, 1.2)

        synthetic_degraded = clean + alpha * residual
        synthetic_degraded = synthetic_degraded.clamp(0, 1)

        return synthetic_degraded

    def __getitem__(self, idx):
        idx = idx % len(self.degraded_files)
        degraded_path = self.degraded_files[idx]
        clean_path = self._get_clean_path(degraded_path)

        degraded = load_rgb(degraded_path)
        clean = load_rgb(clean_path)
        label = self._get_degradation_label(degraded_path)

        if self.split == "train":
            degraded, clean = self._random_crop_pair(degraded, clean)
            degraded, clean = self._augment_pair(degraded, clean)

            degraded = self._apply_same_prompt_residual_aug(
                degraded=degraded,
                clean=clean,
                label=label,
            )

        return degraded, clean, label, degraded_path.name


class TestDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.test_dir = self._find_test_dir(self.data_root)

        self.files = sorted(
            list(self.test_dir.glob("*.png")),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
        )

        print(f"[TestDataset] test_dir = {self.test_dir}")
        print(f"[TestDataset] test images = {len(self.files)}")

        if len(self.files) == 0:
            raise RuntimeError(f"No test png images found in: {self.test_dir}")

    def _find_test_dir(self, root):
        candidates = [
            root / "test" / "degraded",
            root / "Test" / "degraded",
            root / "test",
            root / "Test",
            root / "degraded",
            root,
        ]

        for p in candidates:
            if p.exists() and len(list(p.glob("*.png"))) > 0:
                return p

        degraded_candidates = list(root.rglob("degraded"))
        for p in degraded_candidates:
            png_files = list(p.glob("*.png"))
            if len(png_files) > 0:
                names = [x.name for x in png_files[:5]]
                if any(name[0].isdigit() for name in names):
                    return p

        raise FileNotFoundError(f"Cannot find test degraded folder under: {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = load_rgb(path)
        return img, path.name


# ============================================================
# Full PromptIR Model
# ============================================================
class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps."""
    def __init__(self, num_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = 1e-6

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return (
            x * self.weight.view(1, -1, 1, 1)
            + self.bias.view(1, -1, 1, 1)
        )


class SimpleGate(nn.Module):
    """NAFNet simple gate: split channels and multiply."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    Lightweight restoration block from the NAFNet design.

    It replaces the original Restormer-style MDTA + GDFN block.
    The block keeps two residual branches:
      1) depthwise-conv spatial mixing + simple channel attention
      2) pointwise FFN-like branch

    beta/gamma are zero-initialized residual scales, which makes the
    larger PromptIR-NAF model much easier to train from scratch.
    """
    def __init__(
        self,
        dim,
        dw_expand=2,
        ffn_expand=2,
        dropout_rate=0.0,
        bias=False,
    ):
        super().__init__()

        dw_channel = dim * dw_expand
        ffn_channel = dim * ffn_expand

        self.norm1 = LayerNorm2d(dim)

        self.conv1 = nn.Conv2d(
            dim,
            dw_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.conv2 = nn.Conv2d(
            dw_channel,
            dw_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dw_channel,
            bias=bias,
        )
        self.sg = SimpleGate()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                dw_channel // 2,
                dw_channel // 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
        )

        self.conv3 = nn.Conv2d(
            dw_channel // 2,
            dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.norm2 = LayerNorm2d(dim)
        self.conv4 = nn.Conv2d(
            dim,
            ffn_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.conv5 = nn.Conv2d(
            ffn_channel // 2,
            dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        y = self.dropout1(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg(y)
        y = self.conv5(y)
        y = self.dropout2(y)
        x = x + y * self.gamma

        return x


class TransformerBlock(nn.Module):
    """
    Drop-in NAFBlock replacement.

    The constructor intentionally keeps the old TransformerBlock signature
    so the existing PromptIR encoder/decoder code can stay unchanged.
    num_heads and ffn_expansion are accepted for compatibility only.
    """
    def __init__(self, dim, num_heads=None, ffn_expansion=2.66, bias=False):
        super().__init__()
        self.block = NAFBlock(
            dim=dim,
            dw_expand=2,
            ffn_expand=2,
            dropout_rate=0.0,
            bias=bias,
        )

    def forward(self, x):
        return self.block(x)


class PromptGenBlock(nn.Module):
    """
    PromptIR Prompt Generation Module.
    It learns multiple prompt components and dynamically mixes them
    according to the current image feature.
    """
    def __init__(self, prompt_dim, prompt_len, prompt_size, lin_dim):
        super().__init__()

        self.prompt_param = nn.Parameter(
            torch.randn(
                1,
                prompt_len,
                prompt_dim,
                prompt_size,
                prompt_size,
            ) * 0.02
        )

        self.linear = nn.Linear(lin_dim, prompt_len)

        self.conv = nn.Conv2d(
            prompt_dim,
            prompt_dim,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, x):
        b, c, h, w = x.shape

        emb = x.mean(dim=(-2, -1))
        weights = F.softmax(self.linear(emb), dim=1)

        prompt = (
            weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            * self.prompt_param
        )
        prompt = prompt.sum(dim=1)

        prompt = F.interpolate(
            prompt,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        prompt = self.conv(prompt)
        return prompt


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class PromptIR(nn.Module):
    """
    Full PromptIR / Restormer-style architecture.
    """

    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refine=4,
        heads=(1, 2, 4, 8),
        prompt_len=5,
        ffn_expansion=2.66,
        bias=False,
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            in_ch,
            dim,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        self.enc1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim,
                    heads[0],
                    ffn_expansion=ffn_expansion,
                    bias=bias,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.down1 = Downsample(dim)

        self.enc2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim * 2,
                    heads[1],
                    ffn_expansion=ffn_expansion,
                    bias=bias,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.down2 = Downsample(dim * 2)

        self.enc3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim * 4,
                    heads[2],
                    ffn_expansion=ffn_expansion,
                    bias=bias,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.down3 = Downsample(dim * 4)

        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim * 8,
                    heads[3],
                    ffn_expansion=ffn_expansion,
                    bias=bias,
                )
                for _ in range(num_blocks[3])
            ]
        )

        self.prompt3 = PromptGenBlock(
            prompt_dim=dim * 4,
            prompt_len=prompt_len,
            prompt_size=64,
            lin_dim=dim * 8,
        )

        self.prompt2 = PromptGenBlock(
            prompt_dim=dim * 2,
            prompt_len=prompt_len,
            prompt_size=128,
            lin_dim=dim * 4,
        )

        self.prompt1 = PromptGenBlock(
            prompt_dim=dim,
            prompt_len=prompt_len,
            prompt_size=256,
            lin_dim=dim * 2,
        )

        self.reduce_noise3 = nn.Conv2d(
            dim * 8 + dim * 4,
            dim * 8,
            kernel_size=1,
            bias=bias,
        )

        self.reduce_noise2 = nn.Conv2d(
            dim * 4 + dim * 2,
            dim * 4,
            kernel_size=1,
            bias=bias,
        )

        self.reduce_noise1 = nn.Conv2d(
            dim * 2 + dim,
            dim * 2,
            kernel_size=1,
            bias=bias,
        )

        self.up3 = Upsample(dim * 8)

        self.reduce3 = nn.Conv2d(
            dim * 8,
            dim * 4,
            kernel_size=1,
            bias=bias,
        )

        self.dec3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim * 4,
                    heads[2],
                    ffn_expansion=ffn_expansion,
                    bias=bias,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.up2 = Upsample(dim * 4)

        self.reduce2 = nn.Conv2d(
            dim * 4,
            dim * 2,
            kernel_size=1,
            bias=bias,
        )

        self.dec2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim * 2,
                    heads[1],
                    ffn_expansion=ffn_expansion,
                    bias=bias,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.up1 = Upsample(dim * 2)

        self.dec1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim * 2,
                    heads[0],
                    ffn_expansion=ffn_expansion,
                    bias=bias,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.refine = nn.Sequential(
            *[
                TransformerBlock(
                    dim * 2,
                    heads[0],
                    ffn_expansion=ffn_expansion,
                    bias=bias,
                )
                for _ in range(num_refine)
            ]
        )

        self.output = nn.Conv2d(
            dim * 2,
            out_ch,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

    @staticmethod
    def _pad_to_multiple(x, multiple=8):
        _, _, h, w = x.shape

        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        if pad_h != 0 or pad_w != 0:
            x = F.pad(
                x,
                (0, pad_w, 0, pad_h),
                mode="reflect",
            )

        return x, h, w

    def forward(self, x):
        inp = x

        x, ori_h, ori_w = self._pad_to_multiple(x, multiple=8)

        x0 = self.patch_embed(x)

        e1 = self.enc1(x0)

        e2 = self.down1(e1)
        e2 = self.enc2(e2)

        e3 = self.down2(e2)
        e3 = self.enc3(e3)

        z = self.down3(e3)
        z = self.latent(z)

        p3 = self.prompt3(z)
        z = torch.cat([z, p3], dim=1)
        z = self.reduce_noise3(z)

        d3 = self.up3(z)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.reduce3(d3)
        d3 = self.dec3(d3)

        p2 = self.prompt2(d3)
        d3 = torch.cat([d3, p2], dim=1)
        d3 = self.reduce_noise2(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.reduce2(d2)
        d2 = self.dec2(d2)

        p1 = self.prompt1(d2)
        d2 = torch.cat([d2, p1], dim=1)
        d2 = self.reduce_noise1(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.refine(d1)
        out = self.output(out)

        out = out[:, :, :ori_h, :ori_w]
        out = out + inp

        return out


# ============================================================
# Loss
# ============================================================
def per_sample_reconstruction_loss(pred, target, loss_type):
    if loss_type == "charbonnier":
        diff = pred - target
        loss = torch.sqrt(diff * diff + 1e-6)
        return loss.mean(dim=(1, 2, 3))

    if loss_type == "mse":
        return ((pred - target) ** 2).mean(dim=(1, 2, 3))

    if loss_type == "l1":
        return (pred - target).abs().mean(dim=(1, 2, 3))

    if loss_type == "psnr":
        # Minimizing log(MSE) is equivalent to maximizing PSNR.
        # This can be negative; that is fine for optimization.
        mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
        return 10.0 * torch.log10(mse + 1e-8)

    raise ValueError(f"Unknown loss_type: {loss_type}")


def sobel_edges(x):
    _, c, _, _ = x.shape

    kernel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    kernel_y = torch.tensor(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    kernel_x = kernel_x.repeat(c, 1, 1, 1)
    kernel_y = kernel_y.repeat(c, 1, 1, 1)

    gx = F.conv2d(x, kernel_x, padding=1, groups=c)
    gy = F.conv2d(x, kernel_y, padding=1, groups=c)

    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def per_sample_edge_loss(pred, target):
    pred_edge = sobel_edges(pred)
    target_edge = sobel_edges(target)

    return (pred_edge - target_edge).abs().mean(dim=(1, 2, 3))


def weighted_restoration_loss(
    pred,
    clean,
    labels,
    loss_type,
    rain_loss_weight,
    snow_loss_weight,
    edge_loss_weight,
):
    rec = per_sample_reconstruction_loss(pred, clean, loss_type)

    if edge_loss_weight > 0:
        edge = per_sample_edge_loss(pred, clean)
        rec = rec + edge_loss_weight * edge

    weights = torch.where(
        labels == 0,
        torch.full_like(rec, rain_loss_weight),
        torch.full_like(rec, snow_loss_weight),
    )

    weighted_loss = (rec * weights).mean()
    plain_loss = rec.mean()

    return weighted_loss, plain_loss


# ============================================================
# Visualization
# ============================================================
def save_training_curves(history, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = history["epoch"]

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train total loss")
    plt.plot(epochs, history["train_rec_loss"], label="restoration loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png", dpi=200)
    plt.close()

    valid_indices = [
        i for i, v in enumerate(history["val_psnr"])
        if not np.isnan(v)
    ]

    if len(valid_indices) == 0:
        return

    val_epochs = [history["epoch"][i] for i in valid_indices]
    val_psnr = [history["val_psnr"][i] for i in valid_indices]
    rain_psnr = [history["rain_psnr"][i] for i in valid_indices]
    snow_psnr = [history["snow_psnr"][i] for i in valid_indices]

    plt.figure()
    plt.plot(val_epochs, val_psnr, label="overall PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "psnr_curve.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(val_epochs, rain_psnr, label="rain PSNR")
    plt.plot(val_epochs, snow_psnr, label="snow PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "type_psnr_curve.png", dpi=200)
    plt.close()


@torch.no_grad()
def save_visual_examples(model, val_set, device, save_dir, epoch, num_vis=4):
    model.eval()

    save_dir = Path(save_dir) / "visuals"
    save_dir.mkdir(parents=True, exist_ok=True)

    indices = np.linspace(0, len(val_set) - 1, num_vis).astype(int)

    fig, axes = plt.subplots(num_vis, 4, figsize=(14, 3.5 * num_vis))

    if num_vis == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, idx in enumerate(indices):
        degraded, clean, label, name = val_set[idx]

        x = degraded.unsqueeze(0).to(device)

        pred = model(x)
        pred = pred.clamp(0, 1)

        label_name = "rain" if label == 0 else "snow"
        error_map = (pred[0].cpu() - clean).abs().mean(dim=0)

        axes[row_idx, 0].imshow(chw_to_hwc(degraded))
        axes[row_idx, 0].set_title(f"Degraded\n{name}")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(chw_to_hwc(pred[0]))
        axes[row_idx, 1].set_title("Restored\nFull PromptIR")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(chw_to_hwc(clean))
        axes[row_idx, 2].set_title(f"Clean\ngt={label_name}")
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(error_map.numpy())
        axes[row_idx, 3].set_title("Abs Error")
        axes[row_idx, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / f"epoch_{epoch:03d}.png", dpi=200)
    plt.close()


# ============================================================
# Validation
# ============================================================
@torch.no_grad()
def validate(model, loader, device):
    model.eval()

    all_scores = []
    rain_scores = []
    snow_scores = []

    for degraded, clean, labels, _ in loader:
        degraded = degraded.to(device)
        clean = clean.to(device)
        labels = labels.to(device)

        pred = model(degraded)
        pred = pred.clamp(0, 1)

        psnr = calc_psnr(pred, clean).item()
        all_scores.append(psnr)

        label = int(labels.item())

        if label == 0:
            rain_scores.append(psnr)
        else:
            snow_scores.append(psnr)

    return {
        "overall_psnr": float(np.mean(all_scores)),
        "rain_psnr": float(np.mean(rain_scores)) if rain_scores else 0.0,
        "snow_psnr": float(np.mean(snow_scores)) if snow_scores else 0.0,
    }


# ============================================================
# TTA
# ============================================================
def tta_aug(x, mode):
    if mode < 4:
        return torch.rot90(x, mode, dims=[2, 3])

    k = mode - 4
    x = torch.flip(x, dims=[3])
    x = torch.rot90(x, k, dims=[2, 3])
    return x


def tta_deaug(x, mode):
    if mode < 4:
        return torch.rot90(x, -mode, dims=[2, 3])

    k = mode - 4
    x = torch.rot90(x, -k, dims=[2, 3])
    x = torch.flip(x, dims=[3])
    return x


@torch.no_grad()
def forward_with_tta(model, img):
    preds = []

    for mode in range(8):
        aug_img = tta_aug(img, mode)
        pred = model(aug_img)
        pred = tta_deaug(pred, mode)
        preds.append(pred)

    return torch.stack(preds, dim=0).mean(dim=0)


# ============================================================
# Train
# ============================================================
def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"

    os.makedirs(args.save_dir, exist_ok=True)

    num_blocks = parse_num_blocks(args.num_blocks)

    train_set = RestorationDataset(
        args.data_root,
        split="train",
        val_ratio=args.val_ratio,
        patch_size=args.patch_size,
        seed=args.seed,
        residual_aug_prob=args.residual_aug_prob,
        residual_pool_size=args.residual_pool_size,
        repeat_factor=args.repeat_factor,
    )

    val_set = RestorationDataset(
        args.data_root,
        split="val",
        val_ratio=args.val_ratio,
        patch_size=args.patch_size,
        seed=args.seed,
        residual_aug_prob=0.0,
        residual_pool_size=0,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = PromptIR(
        dim=args.dim,
        num_blocks=num_blocks,
        num_refine=args.num_refine,
        prompt_len=args.prompt_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] PromptIR-NAF(dim={args.dim}, blocks={num_blocks})")
    print(f"[Model] Params: {n_params:.2f}M")

    if args.resume:
        flexible_load_checkpoint(
            model,
            args.resume,
            device,
            use_ema=args.resume_use_ema,
        )

    ema = None

    if args.ema_decay > 0:
        ema = EMA(model, decay=args.ema_decay)
        print(f"[EMA] Enabled. decay={args.ema_decay}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_psnr = -1.0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_rec_loss": [],
        "val_psnr": [],
        "rain_psnr": [],
        "snow_psnr": [],
    }

    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Loss type: {args.loss_type}")
    print(f"Rain loss weight: {args.rain_loss_weight}")
    print(f"Snow loss weight: {args.snow_loss_weight}")
    print(f"Edge loss weight: {args.edge_loss_weight}")
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")
    print(f"Save dir: {args.save_dir}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        losses = []
        rec_losses = []

        for degraded, clean, labels, _ in train_loader:
            degraded = degraded.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(degraded)

                weighted_rec_loss, plain_rec_loss = weighted_restoration_loss(
                    pred=pred,
                    clean=clean,
                    labels=labels,
                    loss_type=args.loss_type,
                    rain_loss_weight=args.rain_loss_weight,
                    snow_loss_weight=args.snow_loss_weight,
                    edge_loss_weight=args.edge_loss_weight,
                )

                loss = weighted_rec_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            losses.append(loss.item())
            rec_losses.append(plain_rec_loss.item())

        scheduler.step()

        avg_loss = float(np.mean(losses))
        avg_rec_loss = float(np.mean(rec_losses))

        val_psnr = np.nan
        rain_psnr = np.nan
        snow_psnr = np.nan

        if epoch % args.val_every == 0 or epoch == 1:
            eval_model = ema.ema_model if ema is not None else model
            val_result = validate(eval_model, val_loader, device)

            val_psnr = val_result["overall_psnr"]
            rain_psnr = val_result["rain_psnr"]
            snow_psnr = val_result["snow_psnr"]

            print(
                f"Epoch [{epoch:03d}/{args.epochs}] "
                f"Loss: {avg_loss:.6f} "
                f"Rec: {avg_rec_loss:.6f} "
                f"PSNR: {val_psnr:.4f} "
                f"Rain: {rain_psnr:.4f} "
                f"Snow: {snow_psnr:.4f}"
            )

            ckpt = {
                "model": model.state_dict(),
                "ema_model": ema.state_dict() if ema is not None else None,
                "epoch": epoch,
                "val_psnr": val_psnr,
                "args": vars(args),
            }

            torch.save(ckpt, Path(args.save_dir) / "last.pth")

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(ckpt, Path(args.save_dir) / "best.pth")
                print(f"Saved best checkpoint. PSNR = {best_psnr:.4f}")

            if epoch % args.visual_every == 0 or epoch == 1:
                save_visual_examples(
                    model=eval_model,
                    val_set=val_set,
                    device=device,
                    save_dir=args.save_dir,
                    epoch=epoch,
                    num_vis=args.num_vis,
                )
        else:
            print(
                f"Epoch [{epoch:03d}/{args.epochs}] "
                f"Loss: {avg_loss:.6f} "
                f"Rec: {avg_rec_loss:.6f}"
            )

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["train_rec_loss"].append(avg_rec_loss)
        history["val_psnr"].append(val_psnr)
        history["rain_psnr"].append(rain_psnr)
        history["snow_psnr"].append(snow_psnr)

        append_log_csv(
            Path(args.save_dir) / "train_log.csv",
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_rec_loss": avg_rec_loss,
                "val_psnr": val_psnr,
                "rain_psnr": rain_psnr,
                "snow_psnr": snow_psnr,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

        save_training_curves(history, args.save_dir)

    print(f"Training finished. Best PSNR: {best_psnr:.4f}")


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {})

    dim = ckpt_args.get("dim", args.dim)
    num_blocks = parse_num_blocks(ckpt_args.get("num_blocks", args.num_blocks))
    num_refine = ckpt_args.get("num_refine", args.num_refine)
    prompt_len = ckpt_args.get("prompt_len", args.prompt_len)

    model = PromptIR(
        dim=dim,
        num_blocks=num_blocks,
        num_refine=num_refine,
        prompt_len=prompt_len,
    ).to(device)

    if args.use_ema and "ema_model" in ckpt and ckpt["ema_model"] is not None:
        print("[Infer] Loading EMA model.")
        model.load_state_dict(ckpt["ema_model"], strict=True)
    else:
        print("[Infer] Loading normal model.")
        model.load_state_dict(ckpt["model"], strict=True)

    model.eval()

    test_set = TestDataset(args.data_root)

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    pred_dict = {}

    for img, name in test_loader:
        img = img.to(device)

        if args.tta:
            pred = forward_with_tta(model, img)
        else:
            pred = model(img)

        pred = pred.clamp(0, 1)

        arr = tensor_to_uint8_chw(pred[0])
        pred_dict[name[0]] = arr

        print(f"Processed {name[0]} -> {arr.shape}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "pred.npz"
    np.savez_compressed(npz_path, **pred_dict)

    zip_path = out_dir / "prediction.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(npz_path, arcname="pred.npz")

    print(f"Saved: {npz_path}")
    print(f"Saved zip for CodaBench: {zip_path}")


# ============================================================
# Main
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "infer"],
    )

    parser.add_argument("--data_root", type=str, required=True)

    parser.add_argument("--save_dir", type=str, default="./runs/full_promptir")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume_use_ema", action="store_true")

    parser.add_argument("--ckpt", type=str, default="./runs/full_promptir/best.pth")
    parser.add_argument("--out_dir", type=str, default="./submission")

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=196)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--val_every", type=int, default=5)

    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--num_blocks", type=str, default="4,6,6,8")
    parser.add_argument("--num_refine", type=int, default=4)
    parser.add_argument("--prompt_len", type=int, default=5)

    parser.add_argument(
        "--loss_type",
        type=str,
        default="charbonnier",
        choices=["charbonnier", "mse", "l1", "psnr"],
    )

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--rain_loss_weight", type=float, default=1.3)
    parser.add_argument("--snow_loss_weight", type=float, default=1.0)
    parser.add_argument("--edge_loss_weight", type=float, default=0.02)

    parser.add_argument("--residual_aug_prob", type=float, default=0.0)
    parser.add_argument("--residual_pool_size", type=int, default=300)
    parser.add_argument("--repeat_factor", type=int, default=1)

    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--visual_every", type=int, default=10)
    parser.add_argument("--num_vis", type=int, default=4)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--use_ema", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(args)

    elif args.mode == "infer":
        infer(args)