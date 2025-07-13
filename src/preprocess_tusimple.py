import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# Paths
root_dir = os.path.join(os.path.dirname(__file__), '..')
tusimple_dir = os.path.join(root_dir, 'TuSimple')
label_paths = [
    os.path.join(tusimple_dir, 'label_data_0313.json'),
    os.path.join(tusimple_dir, 'label_data_0531.json'),
    os.path.join(tusimple_dir, 'label_data_0601.json')
]
output_dir = os.path.join(tusimple_dir, 'processed')

# Create output directories
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

# Global counters
processed_count = 0
skipped_count = 0

# Load and process each JSON file
for label_path in label_paths:
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"Processing {os.path.basename(label_path)}"):
        data = json.loads(line)
        raw_path = data['raw_file']
        lanes = data['lanes']
        h_samples = data['h_samples']

        image_path = os.path.join(tusimple_dir, raw_path)
        image = cv2.imread(image_path)
        if image is None:
            skipped_count += 1
            continue

        # Make filename unique
        relative_path = os.path.relpath(image_path, tusimple_dir)
        filename = relative_path.replace(os.sep, '_')

        # Save image
        cv2.imwrite(os.path.join(output_dir, 'images', filename), image)

        # Create and save mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for lane in lanes:
            points = [(int(x), int(y)) for x, y in zip(lane, h_samples) if x >= 0]
            if len(points) > 1:
                cv2.polylines(mask, [np.array(points)], isClosed=False, color=1, thickness=5)
        cv2.imwrite(os.path.join(output_dir, 'masks', filename), mask)

        processed_count += 1

# Final report
print(f"\n✅ Processed {processed_count} images.")
print(f"❌ Skipped {skipped_count} images (missing or invalid).")
