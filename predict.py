#!/usr/bin/env python3
"""
Battery Position Prediction and Failure Recording Script

This script uses a trained YOLO model to predict battery positions in images
and records detailed failure analysis for model improvement.

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
import shutil
from pathlib import Path
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class BatteryPredictor:
    def __init__(self, model_path, dataset_path, output_dir="prediction_results"):
        """
        Initialize the battery predictor.
        
        Args:
            model_path: Path to trained YOLO model
            dataset_path: Path to dataset directory
            output_dir: Directory to save prediction results
        """
        self.model_path = model_path
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "successful_predictions").mkdir(exist_ok=True)
        (self.output_dir / "failed_predictions").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # Initialize tracking
        self.predictions = []
        self.failures = []
        self.failure_stats = {}
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")
        
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
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes."""
        # Convert YOLO format to corner coordinates
        def yolo_to_corners(box):
            x_center, y_center, width, height = box['x_center'], box['y_center'], box['width'], box['height']
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            return x1, y1, x2, y2
        
        x1_1, y1_1, x2_1, y2_1 = yolo_to_corners(box1)
        x1_2, y1_2, x2_2, y2_2 = yolo_to_corners(box2)
        
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
    
    def find_best_matches(self, ground_truth, predictions, iou_threshold=0.5):
        """Find best matches between ground truth and predictions."""
        matched_gt = []
        matched_pred = []
        unmatched_gt = []
        unmatched_pred = []
        
        # Create copies to avoid modifying original lists
        gt_copy = ground_truth.copy()
        pred_copy = predictions.copy()
        
        # Find matches
        for gt_box in gt_copy[:]:
            best_iou = 0
            best_pred_idx = -1
            
            for i, pred_box in enumerate(pred_copy):
                iou = self.calculate_iou(gt_box, pred_box)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_pred_idx = i
            
            if best_pred_idx != -1:
                matched_gt.append(gt_box)
                matched_pred.append(pred_copy[best_pred_idx])
                pred_copy.pop(best_pred_idx)
            else:
                unmatched_gt.append(gt_box)
        
        unmatched_pred = pred_copy
        
        return matched_gt, matched_pred, unmatched_gt, unmatched_pred
    
    def analyze_prediction(self, image_name, ground_truth, predictions, confidence_threshold=0.25):
        """Analyze a single prediction and determine success/failure."""
        # Filter predictions by confidence threshold
        filtered_predictions = [p for p in predictions if p['confidence'] >= confidence_threshold]
        
        # Find matches
        matched_gt, matched_pred, unmatched_gt, unmatched_pred = self.find_best_matches(
            ground_truth, filtered_predictions
        )
        
        # Calculate metrics
        precision = len(matched_pred) / len(filtered_predictions) if filtered_predictions else 0
        recall = len(matched_gt) / len(ground_truth) if ground_truth else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Determine if prediction is successful
        # Use a more reasonable threshold that aligns with Hub metrics
        success = f1_score > 0.3 or (precision > 0.8 and recall > 0.8)
        
        # Calculate IoU statistics for matched predictions
        ious = []
        if matched_gt and matched_pred:
            for gt, pred in zip(matched_gt, matched_pred):
                iou = self.calculate_iou(gt, pred)
                ious.append(iou)
        
        # Create prediction record
        prediction_record = {
            'image_name': image_name,
            'timestamp': datetime.now().isoformat(),
            'total_gt_objects': len(ground_truth),
            'total_predictions': len(filtered_predictions),
            'matched_objects': len(matched_gt),
            'false_negatives': len(unmatched_gt),
            'false_positives': len(unmatched_pred),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'success': success,
            'avg_confidence': np.mean([p['confidence'] for p in filtered_predictions]) if filtered_predictions else 0,
            'max_confidence': np.max([p['confidence'] for p in filtered_predictions]) if filtered_predictions else 0,
            'min_confidence': np.min([p['confidence'] for p in filtered_predictions]) if filtered_predictions else 0,
            'avg_iou': np.mean(ious) if ious else 0,
            'max_iou': np.max(ious) if ious else 0,
            'min_iou': np.min(ious) if ious else 0
        }
        
        return prediction_record, matched_gt, matched_pred, unmatched_gt, unmatched_pred
    
    def predict_single_image(self, image_path, ground_truth_path=None, confidence_threshold=0.25):
        """Predict battery position in a single image."""
        image_name = Path(image_path).stem
        
        # Load ground truth if available
        ground_truth = []
        if ground_truth_path and os.path.exists(ground_truth_path):
            ground_truth = self.load_yolo_annotations(ground_truth_path)
        
        # Run prediction
        results = self.model.predict(image_path, conf=confidence_threshold, save=False, verbose=False)
        
        # Convert predictions to our format
        predictions = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    # Convert from pixel coordinates to YOLO format
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    img_height, img_width = result.orig_shape
                    
                    # Convert to YOLO format (normalized)
                    x_center = (x1 + x2) / 2 / img_width
                    y_center = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    predictions.append({
                        'class_id': int(box.cls.item()),
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'confidence': float(box.conf.item())
                    })
        
        # Analyze prediction
        prediction_record, matched_gt, matched_pred, unmatched_gt, unmatched_pred = self.analyze_prediction(
            image_name, ground_truth, predictions, confidence_threshold
        )
        
        # Save prediction with visualization
        if results and len(results) > 0:
            result = results[0]
            output_path = self.output_dir / ("successful_predictions" if prediction_record['success'] else "failed_predictions") / f"{image_name}.jpg"
            cv2.imwrite(str(output_path), result.plot())
        
        return prediction_record, predictions
    
    def predict_dataset(self, confidence_threshold=0.25, iou_threshold=0.5):
        """Predict battery positions for entire dataset."""
        print("Starting battery position prediction...")
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
            print(f"Processing {split} split: {len(image_files)} images")
            
            for i, image_path in enumerate(image_files):
                if i % 50 == 0:
                    print(f"  Processing image {i+1}/{len(image_files)}")
                
                image_name = image_path.stem
                label_path = labels_dir / f"{image_name}.txt"
                
                # Predict and analyze
                prediction_record, predictions = self.predict_single_image(
                    str(image_path), 
                    str(label_path) if label_path.exists() else None,
                    confidence_threshold
                )
                prediction_record['split'] = split
                
                self.predictions.append(prediction_record)
                
                # Record failures
                if not prediction_record['success']:
                    failure_record = {
                        'image_name': image_name,
                        'split': split,
                        'failure_type': self.classify_failure(prediction_record),
                        'f1_score': prediction_record['f1_score'],
                        'precision': prediction_record['precision'],
                        'recall': prediction_record['recall'],
                        'false_negatives': prediction_record['false_negatives'],
                        'false_positives': prediction_record['false_positives'],
                        'avg_confidence': prediction_record['avg_confidence']
                    }
                    self.failures.append(failure_record)
                
                total_images += 1
        
        # Calculate overall statistics
        self.calculate_statistics()
        
        print(f"\nPrediction completed! Processed {total_images} images.")
        print(f"Successful predictions: {sum(1 for p in self.predictions if p['success'])}")
        print(f"Failed predictions: {sum(1 for p in self.predictions if not p['success'])}")
        print(f"Overall F1 Score: {self.failure_stats.get('overall_f1', 0):.3f}")
        
        return self.predictions, self.failures
    
    def classify_failure(self, prediction_record):
        """Classify the type of failure."""
        if prediction_record['false_negatives'] > prediction_record['false_positives']:
            return "missed_detections"
        elif prediction_record['false_positives'] > prediction_record['false_negatives']:
            return "false_detections"
        else:
            return "mixed_failures"
    
    def calculate_statistics(self):
        """Calculate overall statistics."""
        if not self.predictions:
            return
        
        df = pd.DataFrame(self.predictions)
        
        self.failure_stats = {
            'total_images': int(len(df)),
            'successful_predictions': int(sum(1 for p in self.predictions if p['success'])),
            'failed_predictions': int(sum(1 for p in self.predictions if not p['success'])),
            'total_gt_objects': int(df['total_gt_objects'].sum()),
            'total_predictions': int(df['total_predictions'].sum()),
            'total_matched': int(df['matched_objects'].sum()),
            'total_false_negatives': int(df['false_negatives'].sum()),
            'total_false_positives': int(df['false_positives'].sum()),
            'overall_precision': float(df['precision'].mean()),
            'overall_recall': float(df['recall'].mean()),
            'overall_f1': float(df['f1_score'].mean()),
            'avg_confidence': float(df['avg_confidence'].mean()),
            'avg_iou': float(df['avg_iou'].mean())
        }
    
    def create_visualizations(self):
        """Create visualizations for prediction analysis."""
        if not self.predictions:
            print("No prediction data to visualize!")
            return
        
        df = pd.DataFrame(self.predictions)
        
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
        if 'split' in df.columns:
            sns.boxplot(data=df, x='split', y='f1_score', ax=axes[0, 2])
            axes[0, 2].set_title('F1 Score by Dataset Split')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Confidence distribution
        axes[1, 0].hist(df['avg_confidence'], bins=30, alpha=0.7, color='lightcoral')
        axes[1, 0].set_xlabel('Average Confidence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Confidence Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # IoU distribution
        axes[1, 1].hist(df['avg_iou'], bins=30, alpha=0.7, color='lightgreen')
        axes[1, 1].set_xlabel('Average IoU')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('IoU Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Success vs Failure
        success_counts = df['success'].value_counts()
        axes[1, 2].pie(success_counts.values, labels=['Failed', 'Successful'], 
                      autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
        axes[1, 2].set_title('Success vs Failure Rate')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'prediction_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Failure Analysis
        if self.failures:
            failure_df = pd.DataFrame(self.failures)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Failure types
            failure_types = failure_df['failure_type'].value_counts()
            axes[0, 0].bar(failure_types.index, failure_types.values, color=['red', 'orange', 'yellow'])
            axes[0, 0].set_title('Failure Types Distribution')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # F1 Score by failure type
            sns.boxplot(data=failure_df, x='failure_type', y='f1_score', ax=axes[0, 1])
            axes[0, 1].set_title('F1 Score by Failure Type')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # False Negatives vs False Positives
            axes[1, 0].scatter(failure_df['false_positives'], failure_df['false_negatives'], 
                              alpha=0.6, c=failure_df['f1_score'], cmap='Reds')
            axes[1, 0].set_xlabel('False Positives')
            axes[1, 0].set_ylabel('False Negatives')
            axes[1, 0].set_title('False Positives vs False Negatives')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Worst performing images
            worst_images = failure_df.nsmallest(10, 'f1_score')
            y_pos = np.arange(len(worst_images))
            axes[1, 1].barh(y_pos, worst_images['f1_score'], color='red', alpha=0.7)
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels(worst_images['image_name'], fontsize=8)
            axes[1, 1].set_xlabel('F1 Score')
            axes[1, 1].set_title('Worst Performing Images')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'visualizations' / 'failure_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_results(self):
        """Save prediction results and statistics."""
        # Save predictions
        if self.predictions:
            df_predictions = pd.DataFrame(self.predictions)
            df_predictions.to_csv(self.output_dir / 'reports' / 'predictions.csv', index=False)
        
        # Save failures
        if self.failures:
            df_failures = pd.DataFrame(self.failures)
            df_failures.to_csv(self.output_dir / 'reports' / 'failures.csv', index=False)
        
        # Save statistics
        with open(self.output_dir / 'reports' / 'statistics.json', 'w') as f:
            json.dump(self.failure_stats, f, indent=2)
        
        # Create summary report
        self.create_summary_report()
        
        print(f"\nResults saved to: {self.output_dir}")
        print("Generated files:")
        print("  - predictions.csv")
        print("  - failures.csv")
        print("  - statistics.json")
        print("  - summary_report.html")
        print("  - visualizations/")
        print("  - successful_predictions/")
        print("  - failed_predictions/")
    
    def create_summary_report(self):
        """Create a comprehensive summary report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Battery Position Prediction Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
                .warning {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .success {{ background-color: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Battery Position Prediction Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Overall Performance Summary</h2>
                <p><strong>Total Images Processed:</strong> {self.failure_stats.get('total_images', 0)}</p>
                <p><strong>Successful Predictions:</strong> {self.failure_stats.get('successful_predictions', 0)}</p>
                <p><strong>Failed Predictions:</strong> {self.failure_stats.get('failed_predictions', 0)}</p>
                <p><strong>Success Rate:</strong> {(self.failure_stats.get('successful_predictions', 0) / max(self.failure_stats.get('total_images', 1), 1) * 100):.1f}%</p>
                
                <h3>Performance Metrics</h3>
                <ul>
                    <li><strong>Overall Precision:</strong> {self.failure_stats.get('overall_precision', 0):.3f}</li>
                    <li><strong>Overall Recall:</strong> {self.failure_stats.get('overall_recall', 0):.3f}</li>
                    <li><strong>Overall F1 Score:</strong> {self.failure_stats.get('overall_f1', 0):.3f}</li>
                    <li><strong>Average Confidence:</strong> {self.failure_stats.get('avg_confidence', 0):.3f}</li>
                    <li><strong>Average IoU:</strong> {self.failure_stats.get('avg_iou', 0):.3f}</li>
                </ul>
            </div>
            
            <h2>Failure Analysis</h2>
            <p><strong>Total Failures:</strong> {len(self.failures)}</p>
        """
        
        if self.failures:
            failure_df = pd.DataFrame(self.failures)
            failure_types = failure_df['failure_type'].value_counts()
            html_content += "<h3>Failure Types:</h3><ul>"
            for failure_type, count in failure_types.items():
                html_content += f"<li><strong>{failure_type}:</strong> {count}</li>"
            html_content += "</ul>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(self.output_dir / 'reports' / 'summary_report.html', 'w') as f:
            f.write(html_content)

def main():
    """Main function to run battery prediction."""
    # Configuration
    model_path = "https://hub.ultralytics.com/models/6LDB83hmaCxMPV7xqea6"  # Your model path
    dataset_path = "github repo/Automated-Detection-Coordinates-of-Lithium-Batteries-in-Opened-Phones-Tablets-/augmented_dataset"  # Your dataset path
    output_dir = "prediction_results"
    confidence_threshold = 0.25
    
    print("Battery Position Prediction System")
    print("="*50)
    
    # Initialize predictor
    predictor = BatteryPredictor(model_path, dataset_path, output_dir)
    
    # Run prediction
    predictions, failures = predictor.predict_dataset(confidence_threshold)
    
    # Create visualizations
    print("\nCreating visualizations...")
    predictor.create_visualizations()
    
    # Save results
    print("\nSaving results...")
    predictor.save_results()
    
    print("\n" + "="*50)
    print("Battery prediction completed!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
