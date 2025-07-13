import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# Paths (modify if necessary)
root_dir = os.path.join(os.path.dirname(__file__), '..')
tusimple_dir = os.path.join(root_dir, 'TuSimple')
label_path = os.path.join(tusimple_dir, 'label_data_0313.json')
output_dir = os.path.join(tusimple_dir, 'processed')

# Create output directories
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

# Load JSON annotation
with open(label_path, 'r') as f:
    lines = f.readlines()

# Iterate over every label entry
for line in tqdm(lines, desc="Processing label_data_0313.json"):
    data = json.loads(line)
    raw_path = data['raw_file']  # e.g., "clips/0313-1/29920/20.jpg"
    lanes = data['lanes']        # list of x positions for each lane
    h_samples = data['h_samples']  # list of y positions

    # Construct full path
    image_path = os.path.join(tusimple_dir, raw_path)

    # Check if image exists
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Could not load: {image_path}")
        continue

    # Save original RGB image
    filename = os.path.basename(image_path)
    cv2.imwrite(os.path.join(output_dir, 'images', filename), image)

    # Create empty binary mask (same height/width as input)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # For each lane
    for lane in lanes:
        points = []
        for x, y in zip(lane, h_samples):
            if x >= 0:
                points.append((int(x), int(y)))
        if len(points) > 1:
            cv2.polylines(mask, [np.array(points)], isClosed=False, color=1, thickness=5)

    # Save mask
    cv2.imwrite(os.path.join(output_dir, 'masks', filename), mask)
