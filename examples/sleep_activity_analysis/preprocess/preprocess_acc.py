import os
import subprocess
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class AccelerometerProcessor:
    def __init__(self, output_dir: str = "accelerometer_results"):
        self.output_dir = output_dir
        self.processed_files = {}
        
    def check_accelerometer_install(self) -> bool:
        try:
            result = subprocess.run(['accProcess', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def install_accelerometer_library(self):
        print("Installing accelerometer library...")
        try:
            subprocess.run(['pip', 'install', 'accelerometer'], check=True)
            print("Accelerometer library installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install accelerometer library: {e}")
            raise
    
    def process_accelerometer_data(self, csv_file: str, 
                                 epoch_period: int = 30,
                                 mg_cutpoint_mvpa: int = 100,
                                 mg_cutpoint_vpa: int = 400) -> Dict[str, str]:
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Input file not found: {csv_file}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_prefix = os.path.join(self.output_dir, base_name)
        
        cmd = [
            'accProcess',
            csv_file,
            '--csvTimeFormat', 'yyyy-MM-dd HH:mm:ss',
            '--csvTimeXYZTempColsIndex', '0,1,2,3',
            '--epochPeriod', str(epoch_period),
            '--outputFolder', self.output_dir
        ]
        
        print(f"Processing {csv_file} with accelerometer library...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Accelerometer processing completed successfully!")
            
            expected_files = {
                'summary': f"{output_prefix}-summary.json",
                'timeseries': f"{output_prefix}-timeSeries.csv.gz",
                'epochs': f"{output_prefix}-epoch.csv",
                'nonwear': f"{output_prefix}-nonWearBouts.csv"
            }
            
            actual_files = {}
            for key, filepath in expected_files.items():
                if os.path.exists(filepath):
                    actual_files[key] = filepath
                    print(f"Created: {filepath}")
                else:
                    print(f"Warning: Expected file not found: {filepath}")
            
            return actual_files
            
        except subprocess.CalledProcessError as e:
            print(f"Error processing accelerometer data: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise
    
    def save_results(self, movement_metrics: pd.DataFrame,
                    activity_summary: Dict[str, Any],
                    pyactigraphy_file: str) -> Dict[str, str]:
        
        movement_file = os.path.join(self.output_dir, 'movement_metrics.csv')
        movement_metrics.to_csv(movement_file, index=False)
        
        summary_file = os.path.join(self.output_dir, 'activity_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(activity_summary, f, indent=2, default=str)
        
        return {
            'movement_metrics': movement_file,
            'activity_summary': summary_file,
            'pyactigraphy_format': pyactigraphy_file
        }
    
    def process_pipeline(self, accelerometer_csv: str) -> Dict[str, Any]:
        
        print("Checking accelerometer library installation...")
        if not self.check_accelerometer_install():
            print("Accelerometer library not found, installing...")
            self.install_accelerometer_library()
        
        print(f"Processing accelerometer data from: {accelerometer_csv}")
        processed_files = self.process_accelerometer_data(accelerometer_csv)
        
        results = {'processed_files': processed_files}
        
        return results

def main():
    processor = AccelerometerProcessor(output_dir="accelerometer_results")
    
    accelerometer_csv = "processed_data/processed_acc_data.csv"
    
    if not os.path.exists(accelerometer_csv):
        print(f"Error: Accelerometer CSV file not found: {accelerometer_csv}")
        print("Please run the orientation preprocessing script first to generate this file.")
        return
    
    try:
        results = processor.process_pipeline(accelerometer_csv)
        
        print("\nGenerated files:")
        if 'output_files' in results:
            for key, filepath in results['output_files'].items():
                print(f"  {key}: {filepath}")
        
        if 'processed_files' in results:
            print("\nAccelerometer library output files:")
            for key, filepath in results['processed_files'].items():
                print(f"  {key}: {filepath}")
                
    except Exception as e:
        print(f"Error during accelerometer processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()