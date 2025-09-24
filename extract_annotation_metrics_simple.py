#!/usr/bin/env python3
"""
YOLO Annotation Metrics Extractor (No Dependencies)
Automatically extracts aspect ratio, area percentage, and centroid from YOLO annotations.
Provides framework for manual tagging of occlusion/adjacency information.
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple

class YOLOAnnotationAnalyzer:
    def __init__(self, dataset_path: str):
        """
        Initialize the analyzer with dataset path.
        
        Args:
            dataset_path: Path to YOLO dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.results = []
        
    def get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """Get image dimensions using PIL or OpenCV."""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.size  # Returns (width, height)
        except ImportError:
            try:
                import cv2
                img = cv2.imread(image_path)
                if img is not None:
                    h, w = img.shape[:2]
                    return (w, h)  # Return (width, height)
            except ImportError:
                pass
            return (0, 0)
    
    def parse_yolo_annotation(self, annotation_path: str, image_width: int, image_height: int) -> List[Dict]:
        """
        Parse YOLO annotation file and extract metrics.
        
        Args:
            annotation_path: Path to .txt annotation file
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            List of dictionaries containing annotation metrics
        """
        annotations = []
        
        if not os.path.exists(annotation_path):
            return annotations
        
        with open(annotation_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    print(f"Warning: Invalid annotation format in {annotation_path}, line {line_num}")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Calculate metrics
                    aspect_ratio = width / height if height > 0 else 0
                    
                    # Convert normalized coordinates to pixel coordinates
                    pixel_width = width * image_width
                    pixel_height = height * image_height
                    pixel_area = pixel_width * pixel_height
                    image_area = image_width * image_height
                    area_percentage = (pixel_area / image_area) * 100 if image_area > 0 else 0
                    
                    # Centroid (already normalized in YOLO format)
                    centroid_x = x_center
                    centroid_y = y_center
                    
                    annotation_data = {
                        'class_id': class_id,
                        'aspect_ratio': aspect_ratio,
                        'area_percentage': area_percentage,
                        'centroid_x': centroid_x,
                        'centroid_y': centroid_y,
                        'width_normalized': width,
                        'height_normalized': height,
                        'width_pixels': pixel_width,
                        'height_pixels': pixel_height,
                        'area_pixels': pixel_area,
                        'image_width': image_width,
                        'image_height': image_height,
                        'image_area': image_area
                    }
                    
                    annotations.append(annotation_data)
                    
                except (ValueError, IndexError) as e:
                    print(f"Error parsing annotation in {annotation_path}, line {line_num}: {e}")
                    continue
        
        return annotations
    
    def process_dataset(self, splits: List[str] = None) -> List[Dict]:
        """
        Process entire dataset and extract metrics.
        
        Args:
            splits: List of dataset splits to process (e.g., ['train', 'val', 'test'])
            
        Returns:
            List of all annotation metrics
        """
        if splits is None:
            splits = ['train', 'val', 'test']
        
        all_annotations = []
        
        for split in splits:
            print(f"Processing {split} split...")
            
            images_dir = self.dataset_path / 'images' / split
            labels_dir = self.dataset_path / 'labels' / split
            
            if not images_dir.exists() or not labels_dir.exists():
                print(f"Warning: {split} split not found, skipping...")
                continue
            
            # Get all image files
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            
            for image_file in image_files:
                # Get corresponding annotation file
                annotation_file = labels_dir / (image_file.stem + '.txt')
                
                if not annotation_file.exists():
                    print(f"Warning: No annotation file for {image_file.name}")
                    continue
                
                # Get image dimensions
                image_width, image_height = self.get_image_dimensions(str(image_file))
                
                if image_width == 0 or image_height == 0:
                    print(f"Warning: Could not get dimensions for {image_file.name}")
                    continue
                
                # Parse annotations
                annotations = self.parse_yolo_annotation(
                    str(annotation_file), 
                    image_width, 
                    image_height
                )
                
                # Add metadata
                for annotation in annotations:
                    annotation.update({
                        'split': split,
                        'image_name': image_file.name,
                        'image_path': str(image_file),
                        'annotation_path': str(annotation_file)
                    })
                
                all_annotations.extend(annotations)
        
        self.results = all_annotations
        return all_annotations
    
    def calculate_statistics(self, data: List[float]) -> Dict:
        """Calculate basic statistics for a list of numbers."""
        if not data:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
        
        data_sorted = sorted(data)
        n = len(data)
        
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / n
        std = variance ** 0.5
        min_val = min(data)
        max_val = max(data)
        median = data_sorted[n // 2] if n % 2 == 1 else (data_sorted[n // 2 - 1] + data_sorted[n // 2]) / 2
        
        return {
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'median': median
        }
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for the dataset."""
        if not self.results:
            return {}
        
        # Count unique images
        unique_images = len(set(r['image_name'] for r in self.results))
        
        # Count by split
        split_counts = {}
        for result in self.results:
            split = result['split']
            split_counts[split] = split_counts.get(split, 0) + 1
        
        # Calculate statistics for aspect ratios
        aspect_ratios = [r['aspect_ratio'] for r in self.results]
        aspect_ratio_stats = self.calculate_statistics(aspect_ratios)
        
        # Calculate statistics for area percentages
        area_percentages = [r['area_percentage'] for r in self.results]
        area_percentage_stats = self.calculate_statistics(area_percentages)
        
        # Calculate centroid statistics
        centroid_x_values = [r['centroid_x'] for r in self.results]
        centroid_y_values = [r['centroid_y'] for r in self.results]
        centroid_x_stats = self.calculate_statistics(centroid_x_values)
        centroid_y_stats = self.calculate_statistics(centroid_y_values)
        
        summary = {
            'total_annotations': len(self.results),
            'unique_images': unique_images,
            'splits': split_counts,
            'aspect_ratio_stats': aspect_ratio_stats,
            'area_percentage_stats': area_percentage_stats,
            'centroid_distribution': {
                'x_mean': centroid_x_stats['mean'],
                'x_std': centroid_x_stats['std'],
                'y_mean': centroid_y_stats['mean'],
                'y_std': centroid_y_stats['std']
            }
        }
        
        return summary
    
    def save_results(self, output_path: str = "annotation_metrics"):
        """Save results to various formats."""
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        if not self.results:
            print("No results to save. Run process_dataset() first.")
            return
        
        # Save as CSV
        csv_path = output_path / "annotation_metrics.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
        print(f"Results saved to: {csv_path}")
        
        # Save as JSON
        json_path = output_path / "annotation_metrics.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {json_path}")
        
        # Save summary statistics
        summary = self.generate_summary_statistics()
        summary_path = output_path / "summary_statistics.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary statistics saved to: {summary_path}")
        
        # Save detailed report
        self.generate_detailed_report(output_path, summary)
    
    def generate_detailed_report(self, output_path: Path, summary: Dict):
        """Generate a detailed HTML report."""
        if not self.results:
            return
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLO Annotation Metrics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>YOLO Annotation Metrics Report</h1>
            
            <div class="summary">
                <h2>Summary Statistics</h2>
                <p><strong>Total Annotations:</strong> {summary['total_annotations']}</p>
                <p><strong>Unique Images:</strong> {summary['unique_images']}</p>
                
                <h3>Split Distribution</h3>
                <ul>
                    {''.join([f'<li>{split}: {count} annotations</li>' for split, count in summary['splits'].items()])}
                </ul>
                
                <h3>Aspect Ratio Statistics</h3>
                <ul>
                    <li>Mean: {summary['aspect_ratio_stats']['mean']:.3f}</li>
                    <li>Std: {summary['aspect_ratio_stats']['std']:.3f}</li>
                    <li>Min: {summary['aspect_ratio_stats']['min']:.3f}</li>
                    <li>Max: {summary['aspect_ratio_stats']['max']:.3f}</li>
                    <li>Median: {summary['aspect_ratio_stats']['median']:.3f}</li>
                </ul>
                
                <h3>Area Percentage Statistics</h3>
                <ul>
                    <li>Mean: {summary['area_percentage_stats']['mean']:.2f}%</li>
                    <li>Std: {summary['area_percentage_stats']['std']:.2f}%</li>
                    <li>Min: {summary['area_percentage_stats']['min']:.2f}%</li>
                    <li>Max: {summary['area_percentage_stats']['max']:.2f}%</li>
                    <li>Median: {summary['area_percentage_stats']['median']:.2f}%</li>
                </ul>
                
                <h3>Centroid Distribution</h3>
                <ul>
                    <li>X Mean: {summary['centroid_distribution']['x_mean']:.3f} ± {summary['centroid_distribution']['x_std']:.3f}</li>
                    <li>Y Mean: {summary['centroid_distribution']['y_mean']:.3f} ± {summary['centroid_distribution']['y_std']:.3f}</li>
                </ul>
            </div>
            
            <h2>Sample Annotations (First 20)</h2>
            <table>
                <tr>
                    <th>Image</th>
                    <th>Class ID</th>
                    <th>Aspect Ratio</th>
                    <th>Area %</th>
                    <th>Centroid X</th>
                    <th>Centroid Y</th>
                    <th>Width (px)</th>
                    <th>Height (px)</th>
                </tr>
        """
        
        # Add sample data to table
        for i, result in enumerate(self.results[:20]):
            html_content += f"""
                <tr>
                    <td>{result['image_name']}</td>
                    <td>{result['class_id']}</td>
                    <td>{result['aspect_ratio']:.3f}</td>
                    <td>{result['area_percentage']:.2f}%</td>
                    <td>{result['centroid_x']:.3f}</td>
                    <td>{result['centroid_y']:.3f}</td>
                    <td>{result['width_pixels']:.1f}</td>
                    <td>{result['height_pixels']:.1f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        report_path = output_path / "annotation_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Detailed report saved to: {report_path}")

def create_manual_tagging_template(output_path: str = "manual_tagging_template.csv"):
    """
    Create a CSV template for manual tagging of occlusion/adjacency information.
    
    Args:
        output_path: Path to save the template CSV file
    """
    template_data = [
        {
            'image_name': 'example_image.jpg',
            'annotation_id': 1,
            'has_connector': 'yes',  # yes/no
            'occlusion_level': 'none',  # none, partial, heavy
            'adjacent_objects': 'battery_compartment',  # free text description
            'notes': 'Battery clearly visible with no obstructions'
        },
        {
            'image_name': 'example_image2.jpg',
            'annotation_id': 1,
            'has_connector': 'no',
            'occlusion_level': 'partial',
            'adjacent_objects': 'cable, circuit_board',
            'notes': 'Battery partially covered by cable'
        }
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if template_data:
            writer = csv.DictWriter(f, fieldnames=template_data[0].keys())
            writer.writeheader()
            writer.writerows(template_data)
    
    print(f"Manual tagging template saved to: {output_path}")
    print("\nTemplate columns:")
    print("- image_name: Name of the image file")
    print("- annotation_id: ID of the annotation (1, 2, 3, etc.)")
    print("- has_connector: 'yes' or 'no' - whether battery has visible connector")
    print("- occlusion_level: 'none', 'partial', or 'heavy' - level of occlusion")
    print("- adjacent_objects: Free text description of nearby objects")
    print("- notes: Additional observations")

def main():
    """Main function to run the annotation metrics extraction."""
    print("YOLO Annotation Metrics Extractor")
    print("=" * 50)
    
    # Initialize analyzer
    dataset_path = "augmented_dataset" # Adjust path as needed
    analyzer = YOLOAnnotationAnalyzer(dataset_path)
    
    # Process dataset
    print("Processing dataset...")
    results = analyzer.process_dataset()
    
    if not results:
        print("No annotations found. Please check your dataset path.")
        return
    
    print(f"Processed {len(results)} annotations from {len(set(r['image_name'] for r in results))} images")
    
    # Generate and save results
    print("\nGenerating results...")
    analyzer.save_results()
    
    # Create manual tagging template
    print("\nCreating manual tagging template...")
    create_manual_tagging_template()
    
    # Print summary
    summary = analyzer.generate_summary_statistics()
    print(f"\nSummary:")
    print(f"Total annotations: {summary['total_annotations']}")
    print(f"Unique images: {summary['unique_images']}")
    print(f"Aspect ratio range: {summary['aspect_ratio_stats']['min']:.3f} - {summary['aspect_ratio_stats']['max']:.3f}")
    print(f"Area percentage range: {summary['area_percentage_stats']['min']:.2f}% - {summary['area_percentage_stats']['max']:.2f}%")
    
    print("\nExtraction completed! Check the 'annotation_metrics' folder for results.")

if __name__ == "__main__":
    main()

