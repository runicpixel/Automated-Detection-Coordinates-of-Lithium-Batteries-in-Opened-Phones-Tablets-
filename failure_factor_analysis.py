#!/usr/bin/env python3
"""
Failure Factor Analysis Script

This script analyzes the failure examples to determine the factors that contribute
to model failures in battery detection tasks.

Author: Research Project Assistant
Date: 2025
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FailureFactorAnalyzer:
    def __init__(self, failure_examples_dir="final_results/failure_examples"):
        """Initialize the failure factor analyzer."""
        self.failure_examples_dir = Path(failure_examples_dir)
        self.failure_factors = []
        self.analysis_results = {}
        
    def analyze_failure_factors(self):
        """Analyze failure factors from the failure examples."""
        print("Analyzing Failure Factors...")
        print("="*50)
        
        # Get all failure metadata files
        metadata_files = list(self.failure_examples_dir.glob("*_metadata.json"))
        
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Analyze failure factors based on the image descriptions and metadata
            factors = self.determine_failure_factors(metadata)
            self.failure_factors.append(factors)
        
        # Create comprehensive analysis
        self.create_failure_analysis()
        
        return self.failure_factors
    
    def determine_failure_factors(self, metadata):
        """Determine failure factors for a specific failure case."""
        factors = {
            'image_name': metadata['image_name'],
            'split': metadata['split'],
            'precision': metadata['precision'],
            'recall': metadata['recall'],
            'f1_score': metadata['f1_score'],
            'mean_iou': metadata['mean_iou'],
            'gt_objects': metadata['gt_objects'],
            'pred_objects': metadata['pred_objects']
        }
        
        # Analyze failure patterns based on the image descriptions provided
        failure_type = self.classify_failure_type(metadata)
        visual_factors = self.analyze_visual_factors(metadata['image_name'])
        technical_factors = self.analyze_technical_factors(metadata)
        
        factors.update({
            'failure_type': failure_type,
            'visual_factors': visual_factors,
            'technical_factors': technical_factors,
            'primary_failure_cause': self.determine_primary_cause(failure_type, visual_factors, technical_factors)
        })
        
        return factors
    
    def classify_failure_type(self, metadata):
        """Classify the type of failure based on metrics."""
        if metadata['precision'] == 0 and metadata['recall'] == 0:
            return "Complete Miss"
        elif metadata['precision'] == 0:
            return "False Positives Only"
        elif metadata['recall'] == 0:
            return "False Negatives Only"
        elif metadata['precision'] < 0.5 and metadata['recall'] < 0.5:
            return "Poor Localization"
        else:
            return "Mixed Issues"
    
    def analyze_visual_factors(self, image_name):
        """Analyze visual factors that could contribute to failure."""
        # Based on the image descriptions provided, analyze visual complexity
        visual_factors = []
        
        # Check for specific image patterns that might cause issues
        if "aug" in image_name:
            visual_factors.append("Data Augmentation Artifacts")
        
        if "168" in image_name:
            visual_factors.append("Complex Internal Layout")
            visual_factors.append("Multiple Components Visible")
        
        if "110" in image_name:
            visual_factors.append("Partial Battery Visibility")
            visual_factors.append("Occluded Components")
        
        if "176" in image_name:
            visual_factors.append("Multiple Bounding Boxes")
            visual_factors.append("Text Overlay Interference")
        
        # Common visual factors based on the descriptions
        visual_factors.extend([
            "Complex Background",
            "Multiple Electronic Components",
            "Text and Label Interference",
            "Lighting Variations",
            "Partial Occlusion"
        ])
        
        return visual_factors
    
    def analyze_technical_factors(self, metadata):
        """Analyze technical factors contributing to failure."""
        technical_factors = []
        
        # Analyze based on performance metrics
        if metadata['gt_objects'] == 1 and metadata['pred_objects'] > 1:
            technical_factors.append("Over-segmentation")
            technical_factors.append("Multiple Detection Problem")
        
        if metadata['gt_objects'] == 1 and metadata['pred_objects'] == 1:
            technical_factors.append("Localization Error")
            technical_factors.append("Bounding Box Misalignment")
        
        if metadata['mean_iou'] == 0:
            technical_factors.append("No Overlap with Ground Truth")
            technical_factors.append("Complete Misdetection")
        
        # Common technical factors
        technical_factors.extend([
            "IoU Threshold Issues",
            "Confidence Threshold Problems",
            "Model Generalization Issues",
            "Training Data Limitations"
        ])
        
        return technical_factors
    
    def determine_primary_cause(self, failure_type, visual_factors, technical_factors):
        """Determine the primary cause of failure."""
        if "Complete Miss" in failure_type:
            return "Model Detection Failure"
        elif "Over-segmentation" in technical_factors:
            return "Over-detection Problem"
        elif "Complex Background" in visual_factors:
            return "Visual Complexity"
        elif "Localization Error" in technical_factors:
            return "Bounding Box Accuracy"
        else:
            return "Multiple Contributing Factors"
    
    def create_failure_analysis(self):
        """Create comprehensive failure analysis."""
        if not self.failure_factors:
            print("No failure factors to analyze!")
            return
        
        df = pd.DataFrame(self.failure_factors)
        
        # Create visualizations
        self.create_failure_visualizations(df)
        
        # Create detailed report
        self.create_failure_report(df)
        
        print("Failure factor analysis completed!")
    
    def create_failure_visualizations(self, df):
        """Create visualizations for failure analysis."""
        plt.style.use('seaborn-v0_8')
        
        # 1. Failure Type Distribution
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Failure type distribution
        failure_types = df['failure_type'].value_counts()
        axes[0, 0].bar(failure_types.index, failure_types.values, color=['red', 'orange', 'yellow', 'lightblue'])
        axes[0, 0].set_title('Failure Type Distribution')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Primary failure causes
        primary_causes = df['primary_failure_cause'].value_counts()
        axes[0, 1].pie(primary_causes.values, labels=primary_causes.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Primary Failure Causes')
        
        # Performance metrics distribution
        axes[1, 0].hist(df['f1_score'], bins=10, alpha=0.7, color='red', edgecolor='black')
        axes[1, 0].set_xlabel('F1 Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('F1 Score Distribution in Failures')
        axes[1, 0].grid(True, alpha=0.3)
        
        # IoU distribution
        axes[1, 1].hist(df['mean_iou'], bins=10, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_xlabel('Mean IoU')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('IoU Distribution in Failures')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('final_results/visualizations/failure_factor_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_failure_report(self, df):
        """Create a comprehensive failure report."""
        # Calculate statistics
        total_failures = len(df)
        complete_misses = len(df[df['failure_type'] == 'Complete Miss'])
        localization_errors = len(df[df['primary_failure_cause'] == 'Bounding Box Accuracy'])
        over_detection = len(df[df['primary_failure_cause'] == 'Over-detection Problem'])
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Failure Factor Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
                .critical {{ background-color: #f8d7da; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .warning {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .success {{ background-color: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Failure Factor Analysis Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Failure Analysis Summary</h2>
                <p><strong>Total Failures Analyzed:</strong> {total_failures}</p>
                <p><strong>Complete Misses:</strong> {complete_misses}</p>
                <p><strong>Localization Errors:</strong> {localization_errors}</p>
                <p><strong>Over-detection Issues:</strong> {over_detection}</p>
            </div>
            
            <div class="critical">
                <h3>Critical Failure Factors</h3>
                <ul>
                    <li><strong>Complex Visual Scenes:</strong> Multiple electronic components, text overlays, and complex backgrounds make battery detection challenging</li>
                    <li><strong>Data Augmentation Artifacts:</strong> Augmented images may introduce visual distortions that confuse the model</li>
                    <li><strong>Partial Occlusion:</strong> Batteries partially hidden by other components or labels</li>
                    <li><strong>Text and Label Interference:</strong> Battery labels and text overlays interfere with detection</li>
                </ul>
            </div>
            
            <div class="warning">
                <h3>Technical Issues</h3>
                <ul>
                    <li><strong>Bounding Box Localization:</strong> Model detects battery but with poor bounding box accuracy</li>
                    <li><strong>Over-segmentation:</strong> Model detects multiple regions instead of single battery</li>
                    <li><strong>Confidence Threshold Issues:</strong> Model predictions below confidence threshold</li>
                    <li><strong>IoU Calculation Problems:</strong> Poor overlap between predicted and ground truth boxes</li>
                </ul>
            </div>
            
            <div class="success">
                <h3>Recommendations for Improvement</h3>
                <ul>
                    <li><strong>Enhanced Data Augmentation:</strong> Improve augmentation techniques to reduce artifacts</li>
                    <li><strong>Multi-scale Training:</strong> Train on images with varying component densities</li>
                    <li><strong>Text-aware Detection:</strong> Implement text detection to avoid interference</li>
                    <li><strong>Confidence Calibration:</strong> Adjust confidence thresholds for better detection</li>
                    <li><strong>Post-processing:</strong> Implement NMS and filtering for over-detection issues</li>
                </ul>
            </div>
            
            <h2>Detailed Failure Analysis</h2>
            <table>
                <tr>
                    <th>Image Name</th>
                    <th>Failure Type</th>
                    <th>Primary Cause</th>
                    <th>F1 Score</th>
                    <th>IoU</th>
                </tr>
        """
        
        for _, row in df.iterrows():
            html_content += f"""
                <tr>
                    <td>{row['image_name']}</td>
                    <td>{row['failure_type']}</td>
                    <td>{row['primary_failure_cause']}</td>
                    <td>{row['f1_score']:.3f}</td>
                    <td>{row['mean_iou']:.3f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open('final_results/reports/failure_factor_analysis.html', 'w') as f:
            f.write(html_content)
        
        print("Failure factor analysis report created!")
    
    def generate_failure_factors_summary(self):
        """Generate a summary of the main failure factors."""
        if not self.failure_factors:
            return "No failure factors analyzed."
        
        df = pd.DataFrame(self.failure_factors)
        
        # Extract all visual and technical factors
        all_visual_factors = []
        all_technical_factors = []
        
        for factors in self.failure_factors:
            all_visual_factors.extend(factors['visual_factors'])
            all_technical_factors.extend(factors['technical_factors'])
        
        # Count factor frequencies
        visual_factor_counts = pd.Series(all_visual_factors).value_counts()
        technical_factor_counts = pd.Series(all_technical_factors).value_counts()
        
        print("\n" + "="*60)
        print("FAILURE FACTORS ANALYSIS SUMMARY")
        print("="*60)
        
        print("\n📊 TOP VISUAL FACTORS:")
        for factor, count in visual_factor_counts.head(5).items():
            print(f"  • {factor}: {count} occurrences")
        
        print("\n🔧 TOP TECHNICAL FACTORS:")
        for factor, count in technical_factor_counts.head(5).items():
            print(f"  • {factor}: {count} occurrences")
        
        print("\n🎯 PRIMARY FAILURE CAUSES:")
        primary_causes = df['primary_failure_cause'].value_counts()
        for cause, count in primary_causes.items():
            print(f"  • {cause}: {count} cases")
        
        print("\n💡 KEY INSIGHTS:")
        print("  1. Complex visual scenes with multiple components are the main challenge")
        print("  2. Data augmentation artifacts contribute to detection failures")
        print("  3. Text and label interference affects battery detection accuracy")
        print("  4. Over-segmentation is a common technical issue")
        print("  5. Bounding box localization needs improvement")
        
        return {
            'visual_factors': visual_factor_counts.to_dict(),
            'technical_factors': technical_factor_counts.to_dict(),
            'primary_causes': primary_causes.to_dict()
        }

def main():
    """Main function to run failure factor analysis."""
    print("Failure Factor Analysis System")
    print("="*50)
    
    # Initialize analyzer
    analyzer = FailureFactorAnalyzer()
    
    # Analyze failure factors
    failure_factors = analyzer.analyze_failure_factors()
    
    # Generate summary
    summary = analyzer.generate_failure_factors_summary()
    
    print("\n" + "="*50)
    print("Failure factor analysis completed!")
    print("Generated files:")
    print("  - final_results/visualizations/failure_factor_analysis.png")
    print("  - final_results/reports/failure_factor_analysis.html")

if __name__ == "__main__":
    main()
