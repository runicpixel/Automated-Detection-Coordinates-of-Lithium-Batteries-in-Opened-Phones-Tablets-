#!/usr/bin/env python3
"""
Annotation Data Visualization Script

This script creates comprehensive heatmaps and histograms from YOLO annotation data.
It analyzes various metrics including aspect ratios, area percentages, centroid distributions,
and provides insights into the dataset characteristics.

Author: Research Project Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AnnotationVisualizer:
    def __init__(self, csv_path='annotation_metrics/annotation_metrics.csv'):
        """Initialize the visualizer with annotation data."""
        self.df = pd.read_csv(csv_path)
        self.setup_plotting()
        
    def setup_plotting(self):
        """Setup matplotlib parameters for better visualization."""
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
    def create_centroid_heatmap(self):
        """Create 2D heatmap of annotation centroids."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Create 2D histogram for centroid distribution
        x_centroids = self.df['centroid_x']
        y_centroids = self.df['centroid_y']
        
        # Plot 1: 2D histogram heatmap
        h = ax1.hist2d(x_centroids, y_centroids, bins=20, cmap='YlOrRd', alpha=0.8)
        ax1.set_xlabel('Centroid X (normalized)')
        ax1.set_ylabel('Centroid Y (normalized)')
        ax1.set_title('Annotation Centroid Distribution Heatmap')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(h[3], ax=ax1, label='Count')
        
        # Plot 2: Scatter plot with density
        scatter = ax2.scatter(x_centroids, y_centroids, c=self.df['area_percentage'], 
                             cmap='viridis', alpha=0.6, s=50)
        ax2.set_xlabel('Centroid X (normalized)')
        ax2.set_ylabel('Centroid Y (normalized)')
        ax2.set_title('Centroid Distribution (colored by area %)')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Area Percentage (%)')
        
        plt.tight_layout()
        plt.savefig('centroid_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_aspect_ratio_heatmap(self):
        """Create heatmap showing aspect ratio distribution by splits."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Aspect ratio histogram
        axes[0, 0].hist(self.df['aspect_ratio'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(self.df['aspect_ratio'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.df["aspect_ratio"].mean():.2f}')
        axes[0, 0].axvline(self.df['aspect_ratio'].median(), color='green', linestyle='--', 
                          label=f'Median: {self.df["aspect_ratio"].median():.2f}')
        axes[0, 0].set_xlabel('Aspect Ratio')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Aspect Ratio Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Aspect ratio by split
        split_data = []
        split_labels = []
        for split in self.df['split'].unique():
            split_data.append(self.df[self.df['split'] == split]['aspect_ratio'].values)
            split_labels.append(f'{split} (n={len(self.df[self.df["split"] == split])})')
        
        axes[0, 1].boxplot(split_data, labels=split_labels)
        axes[0, 1].set_ylabel('Aspect Ratio')
        axes[0, 1].set_title('Aspect Ratio by Dataset Split')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Aspect ratio vs area percentage heatmap
        aspect_bins = pd.cut(self.df['aspect_ratio'], bins=10)
        area_bins = pd.cut(self.df['area_percentage'], bins=10)
        crosstab = pd.crosstab(aspect_bins, area_bins)
        
        sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Aspect Ratio vs Area Percentage Heatmap')
        axes[1, 0].set_xlabel('Area Percentage Bins')
        axes[1, 0].set_ylabel('Aspect Ratio Bins')
        
        # Plot 4: Aspect ratio distribution by split (violin plot)
        sns.violinplot(data=self.df, x='split', y='aspect_ratio', ax=axes[1, 1])
        axes[1, 1].set_title('Aspect Ratio Distribution by Split')
        axes[1, 1].set_xlabel('Dataset Split')
        axes[1, 1].set_ylabel('Aspect Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('aspect_ratio_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_area_percentage_heatmap(self):
        """Create heatmap for area percentage analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Area percentage histogram
        axes[0, 0].hist(self.df['area_percentage'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 0].axvline(self.df['area_percentage'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.df["area_percentage"].mean():.1f}%')
        axes[0, 0].axvline(self.df['area_percentage'].median(), color='green', linestyle='--', 
                          label=f'Median: {self.df["area_percentage"].median():.1f}%')
        axes[0, 0].set_xlabel('Area Percentage (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Area Percentage Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Area percentage by split
        split_data = []
        split_labels = []
        for split in self.df['split'].unique():
            split_data.append(self.df[self.df['split'] == split]['area_percentage'].values)
            split_labels.append(f'{split} (n={len(self.df[self.df["split"] == split])})')
        
        axes[0, 1].boxplot(split_data, labels=split_labels)
        axes[0, 1].set_ylabel('Area Percentage (%)')
        axes[0, 1].set_title('Area Percentage by Dataset Split')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Area percentage vs image dimensions
        scatter = axes[1, 0].scatter(self.df['image_width'], self.df['image_height'], 
                                   c=self.df['area_percentage'], cmap='plasma', alpha=0.6, s=50)
        axes[1, 0].set_xlabel('Image Width (pixels)')
        axes[1, 0].set_ylabel('Image Height (pixels)')
        axes[1, 0].set_title('Image Dimensions (colored by area %)')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='Area Percentage (%)')
        
        # Plot 4: Area percentage distribution by split (violin plot)
        sns.violinplot(data=self.df, x='split', y='area_percentage', ax=axes[1, 1])
        axes[1, 1].set_title('Area Percentage Distribution by Split')
        axes[1, 1].set_xlabel('Dataset Split')
        axes[1, 1].set_ylabel('Area Percentage (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('area_percentage_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_correlation_heatmap(self):
        """Create correlation heatmap between different metrics."""
        # Select numerical columns for correlation
        numerical_cols = ['aspect_ratio', 'area_percentage', 'centroid_x', 'centroid_y', 
                         'width_normalized', 'height_normalized', 'width_pixels', 'height_pixels']
        
        corr_matrix = self.df[numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Matrix of Annotation Metrics', fontsize=16, pad=20)
        
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_split_comparison_heatmap(self):
        """Create heatmap comparing metrics across dataset splits."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Split distribution
        split_counts = self.df['split'].value_counts()
        axes[0, 0].pie(split_counts.values, labels=split_counts.index, autopct='%1.1f%%', 
                      colors=['lightblue', 'lightgreen', 'lightcoral'])
        axes[0, 0].set_title('Dataset Split Distribution')
        
        # Plot 2: Metrics comparison by split
        metrics = ['aspect_ratio', 'area_percentage', 'centroid_x', 'centroid_y']
        split_stats = self.df.groupby('split')[metrics].mean()
        
        im = axes[0, 1].imshow(split_stats.T, cmap='viridis', aspect='auto')
        axes[0, 1].set_xticks(range(len(split_stats.index)))
        axes[0, 1].set_xticklabels(split_stats.index)
        axes[0, 1].set_yticks(range(len(metrics)))
        axes[0, 1].set_yticklabels(metrics)
        axes[0, 1].set_title('Mean Metrics by Split')
        
        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(split_stats.index)):
                text = axes[0, 1].text(j, i, f'{split_stats.iloc[j, i]:.3f}',
                                     ha="center", va="center", color="white", fontweight='bold')
        
        plt.colorbar(im, ax=axes[0, 1], label='Normalized Values')
        
        # Plot 3: Centroid distribution by split
        for split in self.df['split'].unique():
            split_data = self.df[self.df['split'] == split]
            axes[1, 0].scatter(split_data['centroid_x'], split_data['centroid_y'], 
                             label=split, alpha=0.6, s=50)
        
        axes[1, 0].set_xlabel('Centroid X (normalized)')
        axes[1, 0].set_ylabel('Centroid Y (normalized)')
        axes[1, 0].set_title('Centroid Distribution by Split')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Aspect ratio vs area percentage by split
        for split in self.df['split'].unique():
            split_data = self.df[self.df['split'] == split]
            axes[1, 1].scatter(split_data['aspect_ratio'], split_data['area_percentage'], 
                             label=split, alpha=0.6, s=50)
        
        axes[1, 1].set_xlabel('Aspect Ratio')
        axes[1, 1].set_ylabel('Area Percentage (%)')
        axes[1, 1].set_title('Aspect Ratio vs Area Percentage by Split')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('split_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_comprehensive_histograms(self):
        """Create comprehensive histogram analysis."""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        
        # Define metrics and their properties
        metrics = [
            ('aspect_ratio', 'Aspect Ratio', 'skyblue'),
            ('area_percentage', 'Area Percentage (%)', 'lightcoral'),
            ('centroid_x', 'Centroid X (normalized)', 'lightgreen'),
            ('centroid_y', 'Centroid Y (normalized)', 'gold'),
            ('width_normalized', 'Width (normalized)', 'plum'),
            ('height_normalized', 'Height (normalized)', 'lightcyan'),
            ('width_pixels', 'Width (pixels)', 'salmon'),
            ('height_pixels', 'Height (pixels)', 'lightsteelblue'),
            ('image_width', 'Image Width (pixels)', 'wheat')
        ]
        
        for i, (col, title, color) in enumerate(metrics):
            if i < len(axes):
                axes[i].hist(self.df[col], bins=30, alpha=0.7, color=color, edgecolor='black')
                axes[i].axvline(self.df[col].mean(), color='red', linestyle='--', 
                              label=f'Mean: {self.df[col].mean():.2f}')
                axes[i].axvline(self.df[col].median(), color='green', linestyle='--', 
                              label=f'Median: {self.df[col].median():.2f}')
                axes[i].set_xlabel(title)
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'{title} Distribution')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('comprehensive_histograms.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_size_analysis_heatmap(self):
        """Create heatmap analyzing object sizes and positions."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Width vs Height scatter with area coloring
        scatter = axes[0, 0].scatter(self.df['width_pixels'], self.df['height_pixels'], 
                                   c=self.df['area_percentage'], cmap='viridis', alpha=0.6, s=50)
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Height (pixels)')
        axes[0, 0].set_title('Object Size Distribution (colored by area %)')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0], label='Area Percentage (%)')
        
        # Plot 2: Normalized size distribution
        axes[0, 1].scatter(self.df['width_normalized'], self.df['height_normalized'], 
                          c=self.df['aspect_ratio'], cmap='plasma', alpha=0.6, s=50)
        axes[0, 1].set_xlabel('Width (normalized)')
        axes[0, 1].set_ylabel('Height (normalized)')
        axes[0, 1].set_title('Normalized Size Distribution (colored by aspect ratio)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Position heatmap
        x_bins = np.linspace(0, 1, 11)
        y_bins = np.linspace(0, 1, 11)
        h, xedges, yedges = np.histogram2d(self.df['centroid_x'], self.df['centroid_y'], 
                                         bins=[x_bins, y_bins])
        
        im = axes[1, 0].imshow(h.T, origin='lower', extent=[0, 1, 0, 1], cmap='hot', aspect='auto')
        axes[1, 0].set_xlabel('Centroid X (normalized)')
        axes[1, 0].set_ylabel('Centroid Y (normalized)')
        axes[1, 0].set_title('Annotation Position Heatmap')
        plt.colorbar(im, ax=axes[1, 0], label='Count')
        
        # Plot 4: Size vs position analysis
        size_bins = pd.cut(self.df['area_percentage'], bins=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
        position_bins_x = pd.cut(self.df['centroid_x'], bins=3, labels=['Left', 'Center', 'Right'])
        position_bins_y = pd.cut(self.df['centroid_y'], bins=3, labels=['Top', 'Middle', 'Bottom'])
        
        crosstab = pd.crosstab([position_bins_x, position_bins_y], size_bins)
        sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title('Size vs Position Analysis')
        axes[1, 1].set_xlabel('Object Size')
        axes[1, 1].set_ylabel('Position (X, Y)')
        
        plt.tight_layout()
        plt.savefig('size_analysis_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_summary_report(self):
        """Generate a summary report of the dataset characteristics."""
        print("="*60)
        print("ANNOTATION DATASET SUMMARY REPORT")
        print("="*60)
        
        print(f"\nDataset Overview:")
        print(f"  Total Annotations: {len(self.df)}")
        print(f"  Unique Images: {self.df['image_name'].nunique()}")
        print(f"  Dataset Splits: {dict(self.df['split'].value_counts())}")
        
        print(f"\nAspect Ratio Statistics:")
        print(f"  Mean: {self.df['aspect_ratio'].mean():.3f}")
        print(f"  Std: {self.df['aspect_ratio'].std():.3f}")
        print(f"  Min: {self.df['aspect_ratio'].min():.3f}")
        print(f"  Max: {self.df['aspect_ratio'].max():.3f}")
        print(f"  Median: {self.df['aspect_ratio'].median():.3f}")
        
        print(f"\nArea Percentage Statistics:")
        print(f"  Mean: {self.df['area_percentage'].mean():.1f}%")
        print(f"  Std: {self.df['area_percentage'].std():.1f}%")
        print(f"  Min: {self.df['area_percentage'].min():.1f}%")
        print(f"  Max: {self.df['area_percentage'].max():.1f}%")
        print(f"  Median: {self.df['area_percentage'].median():.1f}%")
        
        print(f"\nCentroid Distribution:")
        print(f"  X Mean: {self.df['centroid_x'].mean():.3f} ± {self.df['centroid_x'].std():.3f}")
        print(f"  Y Mean: {self.df['centroid_y'].mean():.3f} ± {self.df['centroid_y'].std():.3f}")
        
        print(f"\nImage Dimensions:")
        print(f"  Width Range: {self.df['image_width'].min()} - {self.df['image_width'].max()} pixels")
        print(f"  Height Range: {self.df['image_height'].min()} - {self.df['image_height'].max()} pixels")
        print(f"  Mean Width: {self.df['image_width'].mean():.0f} pixels")
        print(f"  Mean Height: {self.df['image_height'].mean():.0f} pixels")
        
        print("\n" + "="*60)
        
    def run_all_visualizations(self):
        """Run all visualization methods."""
        print("Generating comprehensive annotation visualizations...")
        
        # Generate summary report
        self.generate_summary_report()
        
        # Create all visualizations
        print("\nCreating centroid heatmap...")
        self.create_centroid_heatmap()
        
        print("Creating aspect ratio analysis...")
        self.create_aspect_ratio_heatmap()
        
        print("Creating area percentage analysis...")
        self.create_area_percentage_heatmap()
        
        print("Creating correlation heatmap...")
        self.create_correlation_heatmap()
        
        print("Creating split comparison analysis...")
        self.create_split_comparison_heatmap()
        
        print("Creating comprehensive histograms...")
        self.create_comprehensive_histograms()
        
        print("Creating size analysis heatmap...")
        self.create_size_analysis_heatmap()
        
        print("\nAll visualizations completed! Check the generated PNG files.")
        print("Generated files:")
        print("  - centroid_heatmap.png")
        print("  - aspect_ratio_analysis.png")
        print("  - area_percentage_analysis.png")
        print("  - correlation_heatmap.png")
        print("  - split_comparison_analysis.png")
        print("  - comprehensive_histograms.png")
        print("  - size_analysis_heatmap.png")

def main():
    """Main function to run the visualization script."""
    try:
        # Initialize the visualizer
        visualizer = AnnotationVisualizer()
        
        # Run all visualizations
        visualizer.run_all_visualizations()
        
    except FileNotFoundError:
        print("Error: annotation_metrics.csv file not found!")
        print("Please ensure the file exists in the annotation_metrics/ directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your data format and try again.")

if __name__ == "__main__":
    main()
