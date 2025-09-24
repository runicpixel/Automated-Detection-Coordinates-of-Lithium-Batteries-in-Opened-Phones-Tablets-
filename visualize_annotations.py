#!/usr/bin/env python3
"""
Visualization script for manual annotation review.
Run this script to view images with their annotations overlaid.
"""

import cv2
import os
import json
from pathlib import Path

def visualize_annotations(image_path, annotation_data, output_dir="visualized_annotations"):
    """Visualize a single image with its annotations."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    h, w = image.shape[:2]
    
    # Draw annotations
    for i, ann in enumerate(annotation_data):
        # Convert normalized coordinates to pixel coordinates
        x_center = int(ann['centroid_x'] * w)
        y_center = int(ann['centroid_y'] * h)
        width = int(ann['width_normalized'] * w)
        height = int(ann['height_normalized'] * h)
        
        # Calculate bounding box corners
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"Battery {i+1}"
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add metrics info
        info = f"AR:{ann['aspect_ratio']:.2f} Area:{ann['area_percentage']:.1f}%"
        cv2.putText(image, info, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save visualized image
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"Visualized image saved: {output_path}")

def main():
    # Load metrics data
    with open('annotation_metrics/annotation_metrics.json', 'r') as f:
        metrics_data = json.load(f)
    
    # Group by image
    image_groups = {}
    for ann in metrics_data:
        image_name = ann['image_name']
        if image_name not in image_groups:
            image_groups[image_name] = []
        image_groups[image_name].append(ann)
    
    # Visualize each image
    for image_name, annotations in image_groups.items():
        # Find the image path
        image_path = annotations[0]['image_path']
        if os.path.exists(image_path):
            visualize_annotations(image_path, annotations)
        else:
            print(f"Image not found: {image_path}")

if __name__ == "__main__":
    main()
