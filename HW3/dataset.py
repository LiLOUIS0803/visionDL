import json
import os

import cv2
import numpy as np
import tifffile
from tqdm import tqdm


def tif_to_coco(train_dir, output_json_path, val_ratio=0.2, seed=42):
    np.random.seed(seed)

    categories = [
        {"id": 1, "name": "class1"},
        {"id": 2, "name": "class2"},
        {"id": 3, "name": "class3"},
        {"id": 4, "name": "class4"},
    ]

    image_folders = sorted([
        f for f in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, f))
    ])

    all_images = []
    all_annotations = []
    ann_id = 1

    for img_id, folder_name in enumerate(tqdm(image_folders), start=1):
        folder_path = os.path.join(train_dir, folder_name)
        image_path = os.path.join(folder_path, "image.tif")

        img_arr = tifffile.imread(image_path)

        if img_arr.ndim == 2:
            height, width = img_arr.shape
        else:
            height, width = img_arr.shape[:2]

        all_images.append({
            "id": img_id,
            "file_name": os.path.join(folder_name, "image.tif"),
            "width": width,
            "height": height,
        })

        for cat_id, cat_name in enumerate(
            ["class1", "class2", "class3", "class4"], start=1,
        ):
            mask_path = os.path.join(folder_path, f"{cat_name}.tif")
            if not os.path.exists(mask_path):
                continue

            mask = tifffile.imread(mask_path)
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids != 0]

            for inst_id in instance_ids:
                binary_mask = (mask == inst_id).astype(np.uint8)

                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )

                segmentation = []
                for contour in contours:
                    if contour.size >= 6:
                        segmentation.append(contour.flatten().tolist())

                if len(segmentation) == 0:
                    continue

                ys, xs = np.where(binary_mask)
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                bbox_w = x_max - x_min + 1
                bbox_h = y_max - y_min + 1
                area = int(binary_mask.sum())

                all_annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "segmentation": segmentation,
                    "bbox": [x_min, y_min, bbox_w, bbox_h],
                    "area": area,
                    "iscrowd": 0,
                })
                ann_id += 1

    n = len(all_images)
    indices = list(range(n))
    np.random.shuffle(indices)
    val_count = int(n * val_ratio)
    val_img_indices = set(indices[:val_count])
    train_img_indices = set(indices[val_count:])

    def build_split(img_indices):
        imgs = [all_images[i] for i in sorted(img_indices)]
        img_id_set = {img["id"] for img in imgs}
        anns = [a for a in all_annotations if a["image_id"] in img_id_set]
        return {
            "images": imgs,
            "annotations": anns,
            "categories": categories,
        }

    train_data = build_split(train_img_indices)
    val_data = build_split(val_img_indices)

    base_dir = os.path.dirname(output_json_path)
    os.makedirs(base_dir, exist_ok=True)
    train_json = os.path.join(base_dir, "train.json")
    val_json = os.path.join(base_dir, "val.json")

    with open(train_json, "w") as f:
        json.dump(train_data, f)
    with open(val_json, "w") as f:
        json.dump(val_data, f)

    print(
        f"Train: {len(train_data['images'])} images, "
        f"{len(train_data['annotations'])} annotations"
    )
    print(
        f"Val:   {len(val_data['images'])} images, "
        f"{len(val_data['annotations'])} annotations"
    )
    print(f"Saved to {train_json} and {val_json}")


if __name__ == "__main__":
    tif_to_coco(
        train_dir="../data/train/",
        output_json_path="../data/annotations/train.json",
    )
