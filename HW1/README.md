# Image Classification with Multi-Scale CBAM and GeM Pooling

## Introduction
This project implements an image classification pipeline in PyTorch for training, inference, and evaluation. The default model is a **Multi-Scale CBAM + ResNetRS-101** classifier, and an alternative **Multi-Scale CBAM + ResNeXt-101** implementation is also included in the code.

Main features:
- Multi-scale feature extraction from intermediate backbone stages
- CBAM (Channel Attention + Spatial Attention) for feature refinement
- GeM pooling for feature aggregation
- Data augmentation using RandomResizedCrop, RandAugment, RandomErasing, MixUp, and CutMix
- Three running modes: `train`, `test`, and `eval`
- Test-time augmentation (TTA) for inference
- Automatic export of training logs, learning curves, and confusion matrices

Expected dataset structure:

```text
DATA_ROOT/
├── train/
│   ├── 0/
│   ├── 1/
│   └── ...
├── val/
│   ├── 0/
│   ├── 1/
│   └── ...
└── test/
    ├── image_0001.jpg
    ├── image_0002.jpg
    └── ...
```

The `train/` and `val/` folders should follow the `torchvision.datasets.ImageFolder` format. Each class should be stored in a separate folder. The `test/` folder should contain unlabeled images.

## Environment Setup
### 1. Create a Python environment
Python **3.10+** is recommended.

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install torch torchvision timm pillow tqdm matplotlib scikit-learn
```

If you want CUDA-enabled PyTorch, install the matching version from the official PyTorch installation guide.

### 3. Prepare the dataset
Place the dataset under a single root directory and make sure it matches the folder structure shown above.

## Usage
The script supports three modes:
- `train`: train the model and save checkpoints
- `test`: run inference on the test set and export predictions
- `eval`: evaluate on the validation set and save the confusion matrix

### Train
```bash
python main.py \
    --mode train \
    --data_root /path/to/dataset \
    --save_dir ./checkpoints \
    --batch_size 16 \
    --epochs 30 \
    --num_workers 4 \
    --lr_backbone 1e-4 \
    --lr_head 5e-4 \
    --min_lr 1e-6 \
    --weight_decay 1e-2 \
    --dropout 0.3 \
    --label_smoothing 0.1 \
    --early_stop_patience 20 \
    --model_name resnetrs101.tf_in1k
```

Training outputs:
- `best_model.pth`
- `last_model.pth`
- `training_log.csv`
- `loss_curve.png`
- `accuracy_curve.png`

### Test / Inference
```bash
python main.py \
    --mode test \
    --data_root /path/to/dataset \
    --save_dir ./checkpoints \
    --checkpoint ./checkpoints/best_model.pth \
    --batch_size 16 \
    --num_workers 4 \
    --model_name resnetrs101.tf_in1k
```

Inference output:
- `prediction.csv`

### Evaluation
```bash
python main.py \
    --mode eval \
    --data_root /path/to/dataset \
    --save_dir ./checkpoints \
    --checkpoint ./checkpoints/best_model.pth \
    --batch_size 16 \
    --num_workers 4 \
    --model_name resnetrs101.tf_in1k
```

Evaluation outputs:
- `confusion_matrix.csv`
- `confusion_matrix.png`



### Preformance Snapshot
![Preformance Snapshot](HW1\snapshot.png)

