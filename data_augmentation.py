#!/usr/bin/env python3
"""
Data Augmentation Script for YOLO Format Dataset using Albumentations
Applies color/brightness adjustments, noise, and geometric transformations while preserving image size.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import yaml
from typing import List, Tuple
import albumentations as A

class YoloDataAugmenter:
    def __init__(self, dataset_path: str, output_path: str, augmentations_per_image: int = 3):
        """
        Initialize the data augmenter.
        
        Args:
            dataset_path: Path to original YOLO dataset
            output_path: Path where augmented dataset will be saved
            augmentations_per_image: Number of augmented versions per original image
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.augmentations_per_image = augmentations_per_image
        
        # Define augmentation pipeline (preserving image size)
        self.transform = A.Compose([
            # Color and brightness adjustments
            A.OneOf([
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            ], p=0.9),
            
            # Noise
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.7),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.7),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.7),
            ], p=0.6),
            
            # Geometric transformations (preserving size)
            A.OneOf([
                A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.8),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.1, 
                    rotate_limit=10, 
                    border_mode=cv2.BORDER_CONSTANT, 
                    value=0, 
                    p=0.8
                ),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.1, 0.1),
                    rotate=(-10, 10),
                    shear=(-5, 5),
                    border_mode=cv2.BORDER_CONSTANT,
                    cval=0,
                    p=0.8
                ),
            ], p=0.8),
            
            # Additional effects
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
            ], p=0.3),
            
            # Random gamma
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def read_yolo_annotation(self, annotation_path: Path) -> Tuple[List[float], List[int]]:
        """
        Read YOLO format annotation file.
        
        Args:
            annotation_path: Path to .txt annotation file
            
        Returns:
            Tuple of (bounding_boxes, class_labels)
        """
        bboxes = []
        class_labels = []
        
        if annotation_path.exists():
            with open(annotation_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            bboxes.append([x_center, y_center, width, height])
                            class_labels.append(class_id)
        
        return bboxes, class_labels
    
    def write_yolo_annotation(self, annotation_path: Path, bboxes: List[List[float]], class_labels: List[int]):
        """
        Write YOLO format annotation file.
        
        Args:
            annotation_path: Path where annotation file will be saved
            bboxes: List of bounding boxes in YOLO format
            class_labels: List of class labels
        """
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(annotation_path, 'w') as f:
            for bbox, class_label in zip(bboxes, class_labels):
                x_center, y_center, width, height = bbox
                f.write(f"{class_label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def augment_image_and_annotations(self, image_path: Path, annotation_path: Path) -> List[Tuple[np.ndarray, List[List[float]], List[int]]]:
        """
        Apply augmentations to image and corresponding annotations.
        
        Args:
            image_path: Path to image file
            annotation_path: Path to annotation file
            
        Returns:
            List of tuples (augmented_image, augmented_bboxes, class_labels)
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return []
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read annotations
        bboxes, class_labels = self.read_yolo_annotation(annotation_path)
        
        augmented_data = []
        
        for i in range(self.augmentations_per_image):
            try:
                if len(bboxes) > 0:
                    # Apply augmentation with bboxes
                    augmented = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_class_labels = augmented['class_labels']
                else:
                    # Apply augmentation without bboxes (for images with no annotations)
                    augmented = self.transform(image=image, bboxes=[], class_labels=[])
                    aug_image = augmented['image']
                    aug_bboxes = []
                    aug_class_labels = []
                
                augmented_data.append((aug_image, aug_bboxes, aug_class_labels))
                
            except Exception as e:
                print(f"Warning: Augmentation failed for {image_path}: {e}")
                continue
        
        return augmented_data
    
    def setup_output_directories(self):
        """Create output directory structure."""
        splits = ['train', 'val', 'test']
        
        for split in splits:
            images_dir = self.output_path / 'images' / split
            labels_dir = self.output_path / 'labels' / split
            
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
    
    def copy_original_data(self):
        """Copy original images and annotations to output directory."""
        splits = ['train', 'val', 'test']
        
        for split in splits:
            # Copy images
            src_images_dir = self.dataset_path / 'images' / split
            dst_images_dir = self.output_path / 'images' / split
            
            if src_images_dir.exists():
                for image_file in src_images_dir.glob('*.jpg'):
                    dst_path = dst_images_dir / image_file.name
                    if not dst_path.exists():
                        shutil.copy2(image_file, dst_path)
                        print(f"Copied original: {image_file.name}")
            
            # Copy labels
            src_labels_dir = self.dataset_path / 'labels' / split
            dst_labels_dir = self.output_path / 'labels' / split
            
            if src_labels_dir.exists():
                for label_file in src_labels_dir.glob('*.txt'):
                    dst_path = dst_labels_dir / label_file.name
                    if not dst_path.exists():
                        shutil.copy2(label_file, dst_path)
    
    def augment_dataset(self):
        """Augment the entire dataset."""
        print("Setting up output directories...")
        self.setup_output_directories()
        
        print("Copying original data...")
        self.copy_original_data()
        
        splits = ['train', 'val', 'test']
        total_augmented = 0
        
        for split in splits:
            print(f"\\nProcessing {split} split...")
            
            images_dir = self.dataset_path / 'images' / split
            labels_dir = self.dataset_path / 'labels' / split
            
            output_images_dir = self.output_path / 'images' / split
            output_labels_dir = self.output_path / 'labels' / split
            
            if not images_dir.exists():
                print(f"Warning: {images_dir} does not exist, skipping...")
                continue
            
            image_files = list(images_dir.glob('*.jpg'))
            print(f"Found {len(image_files)} images in {split} split")
            
            for i, image_path in enumerate(image_files):
                # Get corresponding annotation file
                annotation_path = labels_dir / (image_path.stem + '.txt')
                
                print(f"Processing {image_path.name} ({i+1}/{len(image_files)})...")
                
                # Generate augmented images
                augmented_data = self.augment_image_and_annotations(image_path, annotation_path)
                
                for j, (aug_image, aug_bboxes, aug_class_labels) in enumerate(augmented_data):
                    # Save augmented image
                    aug_image_name = f"{image_path.stem}_aug_{j+1}.jpg"
                    aug_image_path = output_images_dir / aug_image_name
                    
                    # Convert back to BGR for saving
                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(aug_image_path), aug_image_bgr)
                    
                    # Save augmented annotations
                    aug_annotation_name = f"{image_path.stem}_aug_{j+1}.txt"
                    aug_annotation_path = output_labels_dir / aug_annotation_name
                    
                    self.write_yolo_annotation(aug_annotation_path, aug_bboxes, aug_class_labels)
                    
                    total_augmented += 1
            
            print(f"Completed {split} split")
        
        print(f"\\nAugmentation complete! Generated {total_augmented} augmented images.")
        
        # Copy data.yaml file
        src_yaml = self.dataset_path / 'data.yaml'
        dst_yaml = self.output_path / 'data.yaml'
        if src_yaml.exists():
            shutil.copy2(src_yaml, dst_yaml)
            print("Copied data.yaml file")
        
        print(f"\\nAugmented dataset saved to: {self.output_path}")


def main():
    """Main function to run data augmentation."""
    # Configuration
    dataset_path = "D:/tri3/Research project/dataset"
    output_path = "D:/tri3/Research project/augmented_dataset"
    augmentations_per_image = 3  # Generate 3 augmented versions per original image
    
    print("=" * 60)
    print("YOLO Dataset Augmentation with Albumentations")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"Augmentations per image: {augmentations_per_image}")
    print("\\nAugmentation types:")
    print("✓ Color & Brightness adjustments (ColorJitter, HSV, Brightness/Contrast)")
    print("✓ Noise (Gaussian, ISO, Multiplicative)")
    print("✓ Geometric transformations (Rotation, Scale, Translate, Shear)")
    print("✓ Blur effects (Gaussian, Motion, Regular blur)")
    print("✓ Gamma correction")
    print("✓ Preserves original image size")
    print("\\n" + "=" * 60)
    
    # Initialize augmenter
    augmenter = YoloDataAugmenter(
        dataset_path=dataset_path,
        output_path=output_path,
        augmentations_per_image=augmentations_per_image
    )
    
    # Run augmentation
    augmenter.augment_dataset()
    
    print("\\n" + "=" * 60)
    print("✅ Data augmentation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
