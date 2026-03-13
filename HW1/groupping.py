import os
import csv
import shutil

csv_path = "prediction.csv"
test_dir = "data/test"
output_dir = "output"

os.makedirs(output_dir, exist_ok=True)

with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)

    for row in reader:
        image_name = row["image_name"]
        label = row["pred_label"]

        src = os.path.join(test_dir, image_name + ".jpg")
        dst_dir = os.path.join(output_dir, label)

        os.makedirs(dst_dir, exist_ok=True)

        dst = os.path.join(dst_dir, image_name + ".jpg")

        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print("Missing:", src)