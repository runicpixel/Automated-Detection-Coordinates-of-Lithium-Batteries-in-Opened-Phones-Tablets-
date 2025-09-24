#!/usr/bin/env python3
"""
Final Results Analysis Script

This script generates:
1. Table of final results (mAP@0.5, precision, recall, mean IoU)
2. 10 success examples with correct bounding boxes
3. 10 failure examples with wrong/missed boxes
4. All images show both predicted and ground truth boxes

Author: Research Project Assistant
Date: 2025
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from pathlib import Path
import warnings
from datetime import datetime
import matplotlib.patches as patches
warnings.filterwarnings('ignore')

class FinalResultsAnalyzer:
    def __init__(self, model_path, dataset_path, output_dir="final_results"):
        """Initialize the final results analyzer."""
        self.model_path = model_path
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "success_examples").mkdir(exist_ok=True)
        (self.output_dir / "failure_examples").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")
        
        # Initialize tracking
        self.all_results = []
        self.success_examples = []
        self.failure_examples = []
        
    def load_yolo_annotations(self, annotation_path):
        """Load YOLO format ground truth annotations."""
        annotations = []
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        annotations.append({
                            'class_id': int(parts[0]),
                            'x_center': float(parts[1]),
                            'y_center': float(parts[2]),
                            'width': float(parts[3]),
                            'height': float(parts[4])
                        })
        return annotations
    
    def yolo_to_xyxy(self, yolo_box, img_width, img_height):
        """Convert YOLO format to xyxy format."""
        x_center, y_center, width, height = yolo_box['x_center'], yolo_box['y_center'], yolo_box['width'], yolo_box['height']
        x1 = (x_center - width / 2) * img_width
        y1 = (y_center - height / 2) * img_height
        x2 = (x_center + width / 2) * img_width
        y2 = (y_center + height / 2) * img_height
        return [x1, y1, x2, y2]
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_metrics(self, ground_truth, predictions, iou_threshold=0.5):
        """Calculate precision, recall, F1, and mean IoU."""
        if not ground_truth and not predictions:
            return 1.0, 1.0, 1.0, 1.0  # Perfect if no ground truth and no predictions
        
        if not ground_truth:
            return 0.0, 1.0, 0.0, 0.0  # No ground truth, so recall is perfect but precision is 0
        
        if not predictions:
            return 1.0, 0.0, 0.0, 0.0  # No predictions, so precision is perfect but recall is 0
        
        # Find matches
        matched_gt = set()
        matched_pred = set()
        ious = []
        
        for i, gt_box in enumerate(ground_truth):
            best_iou = 0
            best_pred_idx = -1
            
            for j, pred_box in enumerate(predictions):
                iou = self.calculate_iou(gt_box, pred_box)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_pred_idx = j
            
            if best_pred_idx != -1:
                matched_gt.add(i)
                matched_pred.add(best_pred_idx)
                ious.append(best_iou)
        
        # Calculate metrics
        tp = len(matched_gt)  # True positives
        fp = len(predictions) - len(matched_pred)  # False positives
        fn = len(ground_truth) - len(matched_gt)  # False negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mean_iou = np.mean(ious) if ious else 0
        
        return precision, recall, f1, mean_iou
    
    def draw_boxes_on_image(self, image, gt_boxes, pred_boxes, image_name, is_success=True):
        """Draw both ground truth and predicted boxes on image."""
        img_height, img_width = image.shape[:2]
        
        # Create a copy of the image
        img_with_boxes = image.copy()
        
        # Draw ground truth boxes in green
        for gt_box in gt_boxes:
            x1, y1, x2, y2 = gt_box
            cv2.rectangle(img_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.putText(img_with_boxes, 'GT', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw predicted boxes in red
        for pred_box in pred_boxes:
            x1, y1, x2, y2 = pred_box
            cv2.rectangle(img_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.putText(img_with_boxes, 'PRED', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add title
        title = f"{'SUCCESS' if is_success else 'FAILURE'}: {image_name}"
        cv2.putText(img_with_boxes, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return img_with_boxes
    
    def analyze_dataset(self, confidence_threshold=0.25, iou_threshold=0.5):
        """Analyze the entire dataset and collect examples."""
        print("Starting comprehensive dataset analysis...")
        print("="*60)
        
        # Get all splits
        splits = ['train', 'val', 'test']
        total_images = 0
        
        for split in splits:
            labels_dir = self.dataset_path / 'labels' / split
            images_dir = self.dataset_path / 'images' / split
            
            if not labels_dir.exists() or not images_dir.exists():
                print(f"Skipping {split} split - directory not found")
                continue
                
            # Get all image files
            image_files = list(images_dir.glob('*.jpg'))
            print(f"Analyzing {split} split: {len(image_files)} images")
            
            for i, image_path in enumerate(image_files):
                if i % 50 == 0:
                    print(f"  Processing image {i+1}/{len(image_files)}")
                
                image_name = image_path.stem
                label_path = labels_dir / f"{image_name}.txt"
                
                # Load ground truth
                ground_truth = self.load_yolo_annotations(str(label_path))
                
                # Run prediction
                results = self.model.predict(str(image_path), conf=confidence_threshold, save=False, verbose=False)
                
                # Convert predictions to xyxy format
                predictions = []
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            predictions.append([x1, y1, x2, y2])
                
                # Convert ground truth to xyxy format
                gt_boxes = []
                if ground_truth:
                    img_height, img_width = result.orig_shape if results and len(results) > 0 else (640, 640)
                    for gt in ground_truth:
                        gt_boxes.append(self.yolo_to_xyxy(gt, img_width, img_height))
                
                # Calculate metrics
                precision, recall, f1, mean_iou = self.calculate_metrics(gt_boxes, predictions, iou_threshold)
                
                # Create result record
                result_record = {
                    'image_name': image_name,
                    'split': split,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'mean_iou': mean_iou,
                    'gt_objects': len(gt_boxes),
                    'pred_objects': len(predictions),
                    'gt_boxes': gt_boxes,
                    'pred_boxes': predictions,
                    'image_path': str(image_path)
                }
                
                self.all_results.append(result_record)
                
                # Collect success examples (F1 > 0.8 and IoU > 0.5)
                if f1 > 0.8 and mean_iou > 0.5 and len(self.success_examples) < 10:
                    self.success_examples.append(result_record)
                
                # Collect failure examples (F1 < 0.5 or IoU < 0.3)
                if (f1 < 0.5 or mean_iou < 0.3) and len(self.failure_examples) < 10:
                    self.failure_examples.append(result_record)
                
                total_images += 1
        
        print(f"Analysis completed! Processed {total_images} images")
        print(f"Found {len(self.success_examples)} success examples")
        print(f"Found {len(self.failure_examples)} failure examples")
        
        return self.all_results
    
    def generate_final_results_table(self):
        """Generate the final results table."""
        if not self.all_results:
            print("No results to generate table!")
            return
        
        df = pd.DataFrame(self.all_results)
        
        # Calculate overall metrics
        overall_precision = df['precision'].mean()
        overall_recall = df['recall'].mean()
        overall_f1 = df['f1_score'].mean()
        overall_iou = df['mean_iou'].mean()
        
        # Calculate mAP@0.5 (approximation using F1 scores > 0.5)
        map_50 = len(df[df['f1_score'] > 0.5]) / len(df)
        
        # Create results table
        results_table = {
            'Metric': ['mAP@0.5', 'Precision', 'Recall', 'Mean IoU'],
            'Value': [f"{map_50:.3f}", f"{overall_precision:.3f}", f"{overall_recall:.3f}", f"{overall_iou:.3f}"],
            'Percentage': [f"{map_50*100:.1f}%", f"{overall_precision*100:.1f}%", f"{overall_recall*100:.1f}%", f"{overall_iou*100:.1f}%"]
        }
        
        results_df = pd.DataFrame(results_table)
        
        # Save table
        results_df.to_csv(self.output_dir / 'reports' / 'final_results_table.csv', index=False)
        
        # Create HTML table
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Final Results Table</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .metric {{ background-color: #e8f4fd; }}
                .value {{ background-color: #d4edda; }}
                .percentage {{ background-color: #fff3cd; }}
            </style>
        </head>
        <body>
            <h1>Final Results Table</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Percentage</th>
                </tr>
                <tr>
                    <td class="metric">mAP@0.5</td>
                    <td class="value">{map_50:.3f}</td>
                    <td class="percentage">{map_50*100:.1f}%</td>
                </tr>
                <tr>
                    <td class="metric">Precision</td>
                    <td class="value">{overall_precision:.3f}</td>
                    <td class="percentage">{overall_precision*100:.1f}%</td>
                </tr>
                <tr>
                    <td class="metric">Recall</td>
                    <td class="value">{overall_recall:.3f}</td>
                    <td class="percentage">{overall_recall*100:.1f}%</td>
                </tr>
                <tr>
                    <td class="metric">Mean IoU</td>
                    <td class="value">{overall_iou:.3f}</td>
                    <td class="percentage">{overall_iou*100:.1f}%</td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        with open(self.output_dir / 'reports' / 'final_results_table.html', 'w') as f:
            f.write(html_content)
        
        print("Final results table generated!")
        print(f"mAP@0.5: {map_50:.3f}")
        print(f"Precision: {overall_precision:.3f}")
        print(f"Recall: {overall_recall:.3f}")
        print(f"Mean IoU: {overall_iou:.3f}")
        
        return results_df
    
    def save_success_examples(self):
        """Save 10 success examples with bounding boxes."""
        print(f"Saving {len(self.success_examples)} success examples...")
        
        for i, example in enumerate(self.success_examples):
            # Load image
            image = cv2.imread(example['image_path'])
            if image is None:
                continue
            
            # Draw boxes
            img_with_boxes = self.draw_boxes_on_image(
                image, 
                example['gt_boxes'], 
                example['pred_boxes'], 
                example['image_name'], 
                is_success=True
            )
            
            # Save image
            output_path = self.output_dir / 'success_examples' / f"success_{i+1:02d}_{example['image_name']}.jpg"
            cv2.imwrite(str(output_path), img_with_boxes)
            
            # Save metadata
            metadata = {
                'image_name': example['image_name'],
                'split': example['split'],
                'precision': float(example['precision']),
                'recall': float(example['recall']),
                'f1_score': float(example['f1_score']),
                'mean_iou': float(example['mean_iou']),
                'gt_objects': int(example['gt_objects']),
                'pred_objects': int(example['pred_objects'])
            }
            
            with open(self.output_dir / 'success_examples' / f"success_{i+1:02d}_{example['image_name']}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Success examples saved to: {self.output_dir / 'success_examples'}")
    
    def save_failure_examples(self):
        """Save 10 failure examples with bounding boxes."""
        print(f"Saving {len(self.failure_examples)} failure examples...")
        
        for i, example in enumerate(self.failure_examples):
            # Load image
            image = cv2.imread(example['image_path'])
            if image is None:
                continue
            
            # Draw boxes
            img_with_boxes = self.draw_boxes_on_image(
                image, 
                example['gt_boxes'], 
                example['pred_boxes'], 
                example['image_name'], 
                is_success=False
            )
            
            # Save image
            output_path = self.output_dir / 'failure_examples' / f"failure_{i+1:02d}_{example['image_name']}.jpg"
            cv2.imwrite(str(output_path), img_with_boxes)
            
            # Save metadata
            metadata = {
                'image_name': example['image_name'],
                'split': example['split'],
                'precision': float(example['precision']),
                'recall': float(example['recall']),
                'f1_score': float(example['f1_score']),
                'mean_iou': float(example['mean_iou']),
                'gt_objects': int(example['gt_objects']),
                'pred_objects': int(example['pred_objects'])
            }
            
            with open(self.output_dir / 'failure_examples' / f"failure_{i+1:02d}_{example['image_name']}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Failure examples saved to: {self.output_dir / 'failure_examples'}")
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        if not self.all_results:
            print("No results to visualize!")
            return
        
        df = pd.DataFrame(self.all_results)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Performance Overview
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # F1 Score distribution
        axes[0, 0].hist(df['f1_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(df['f1_score'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["f1_score"].mean():.3f}')
        axes[0, 0].set_xlabel('F1 Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('F1 Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision vs Recall
        axes[0, 1].scatter(df['precision'], df['recall'], alpha=0.6, c=df['f1_score'], cmap='viridis')
        axes[0, 1].set_xlabel('Precision')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_title('Precision vs Recall (colored by F1)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Performance by split
        sns.boxplot(data=df, x='split', y='f1_score', ax=axes[0, 2])
        axes[0, 2].set_title('F1 Score by Dataset Split')
        axes[0, 2].grid(True, alpha=0.3)
        
        # IoU distribution
        axes[1, 0].hist(df['mean_iou'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].axvline(df['mean_iou'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["mean_iou"].mean():.3f}')
        axes[1, 0].set_xlabel('Mean IoU')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('IoU Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance categories
        performance_categories = []
        for f1 in df['f1_score']:
            if f1 >= 0.9:
                performance_categories.append('Excellent (≥0.9)')
            elif f1 >= 0.8:
                performance_categories.append('Good (0.8-0.9)')
            elif f1 >= 0.5:
                performance_categories.append('Fair (0.5-0.8)')
            else:
                performance_categories.append('Poor (<0.5)')
        
        category_counts = pd.Series(performance_categories).value_counts()
        axes[1, 1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Performance Categories')
        
        # Worst performing images
        worst_images = df.nsmallest(10, 'f1_score')
        y_pos = np.arange(len(worst_images))
        axes[1, 2].barh(y_pos, worst_images['f1_score'], color='red', alpha=0.7)
        axes[1, 2].set_yticks(y_pos)
        axes[1, 2].set_yticklabels(worst_images['image_name'], fontsize=8)
        axes[1, 2].set_xlabel('F1 Score')
        axes[1, 2].set_title('Worst Performing Images')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'performance_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Detailed Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Precision distribution by split
        sns.boxplot(data=df, x='split', y='precision', ax=axes[0, 0])
        axes[0, 0].set_title('Precision by Dataset Split')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Recall distribution by split
        sns.boxplot(data=df, x='split', y='recall', ax=axes[0, 1])
        axes[0, 1].set_title('Recall by Dataset Split')
        axes[0, 1].grid(True, alpha=0.3)
        
        # IoU by split
        sns.boxplot(data=df, x='split', y='mean_iou', ax=axes[1, 0])
        axes[1, 0].set_title('IoU by Dataset Split')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Success vs Failure rate
        success_rate = len(df[df['f1_score'] > 0.5]) / len(df)
        failure_rate = 1 - success_rate
        axes[1, 1].pie([success_rate, failure_rate], labels=['Success', 'Failure'], 
                      autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        axes[1, 1].set_title('Success vs Failure Rate')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations created and saved!")
    
    def create_summary_report(self):
        """Create a comprehensive summary report."""
        if not self.all_results:
            print("No results to create summary!")
            return
        
        df = pd.DataFrame(self.all_results)
        
        # Calculate statistics
        total_images = len(df)
        success_count = len(self.success_examples)
        failure_count = len(self.failure_examples)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Final Results Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
                .success {{ background-color: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .failure {{ background-color: #f8d7da; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Final Results Summary Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Overall Statistics</h2>
                <p><strong>Total Images Analyzed:</strong> {total_images}</p>
                <p><strong>Success Examples Collected:</strong> {success_count}</p>
                <p><strong>Failure Examples Collected:</strong> {failure_count}</p>
            </div>
            
            <div class="success">
                <h3>Success Examples</h3>
                <p>These are examples where the model performed well (F1 > 0.8 and IoU > 0.5).</p>
                <p>Green boxes = Ground Truth, Red boxes = Predictions</p>
            </div>
            
            <div class="failure">
                <h3>Failure Examples</h3>
                <p>These are examples where the model struggled (F1 < 0.5 or IoU < 0.3).</p>
                <p>Green boxes = Ground Truth, Red boxes = Predictions</p>
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / 'reports' / 'summary_report.html', 'w') as f:
            f.write(html_content)
    
    def run_complete_analysis(self, confidence_threshold=0.25, iou_threshold=0.5):
        """Run the complete analysis."""
        print("Starting Final Results Analysis...")
        print("="*60)
        
        # Analyze dataset
        self.analyze_dataset(confidence_threshold, iou_threshold)
        
        # Generate final results table
        print("\nGenerating final results table...")
        self.generate_final_results_table()
        
        # Save success examples
        print("\nSaving success examples...")
        self.save_success_examples()
        
        # Save failure examples
        print("\nSaving failure examples...")
        self.save_failure_examples()
        
        # Create visualizations
        print("\nCreating visualizations...")
        self.create_visualizations()
        
        # Create summary report
        print("\nCreating summary report...")
        self.create_summary_report()
        
        print("\n" + "="*60)
        print("Final results analysis completed!")
        print(f"Results saved to: {self.output_dir}")
        print("Generated files:")
        print("  - reports/final_results_table.csv")
        print("  - reports/final_results_table.html")
        print("  - reports/summary_report.html")
        print("  - visualizations/performance_overview.png")
        print("  - visualizations/detailed_analysis.png")
        print("  - success_examples/ (10 success examples)")
        print("  - failure_examples/ (10 failure examples)")

def main():
    """Main function to run the final results analysis."""
    # Configuration
    model_path = "https://hub.ultralytics.com/models/6LDB83hmaCxMPV7xqea6"
    dataset_path = "github repo/Automated-Detection-Coordinates-of-Lithium-Batteries-in-Opened-Phones-Tablets-/augmented_dataset"
    output_dir = "final_results"
    confidence_threshold = 0.25
    iou_threshold = 0.5
    
    print("Final Results Analysis System")
    print("="*50)
    
    # Initialize analyzer
    analyzer = FinalResultsAnalyzer(model_path, dataset_path, output_dir)
    
    # Run complete analysis
    analyzer.run_complete_analysis(confidence_threshold, iou_threshold)

if __name__ == "__main__":
    main()
