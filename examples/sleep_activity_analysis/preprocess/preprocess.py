import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class OrientationPreprocessor:
    def __init__(self, sampling_rate: float = 25.0):
        self.sampling_rate = sampling_rate
        self.epoch_length = 30.0
        self.samples_per_epoch = int(sampling_rate * self.epoch_length)
        
    def load_orientation_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        
        required_cols = ['TimeStamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df['TimeStamp'] = pd.to_numeric(df['TimeStamp'], errors='coerce')
        df = df.dropna(subset=['TimeStamp'])
        df = df.sort_values('TimeStamp').reset_index(drop=True)
        
        return df
    
    def load_eeg_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        
        if 'TimeStamp' not in df.columns:
            raise ValueError("EEG data must contain 'TimeStamp' column")
        
        df['TimeStamp'] = pd.to_numeric(df['TimeStamp'], errors='coerce')
        df = df.dropna(subset=['TimeStamp'])
        df = df.sort_values('TimeStamp').reset_index(drop=True)
        
        return df
    
    def convert_timestamp_to_datetime(self, timestamp: float, start_date: str = "2024-01-01") -> datetime:
        base_date = datetime.strptime(start_date, "%Y-%m-%d")
        return base_date + timedelta(seconds=timestamp)
    
    def create_accelerometer_csv(self, orientation_df: pd.DataFrame, output_path: str, 
                                start_date: str = "2024-01-01") -> str:
        accel_data = orientation_df[['TimeStamp', 'ax', 'ay', 'az']].copy()
        
        accel_data['datetime'] = accel_data['TimeStamp'].apply(
            lambda x: self.convert_timestamp_to_datetime(x, start_date)
        )
        
        accel_data['time'] = accel_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        accel_data['time'] = accel_data['time'].str[:-3]
        
        output_df = accel_data[['time', 'ax', 'ay', 'az']].copy()
        output_df.columns = ['time', 'x', 'y', 'z']
        
        output_df['x'] = output_df['x'] / 1000.0
        output_df['y'] = output_df['y'] / 1000.0
        output_df['z'] = output_df['z'] / 1000.0
        
        output_df.to_csv(output_path, index=False)
        return output_path
    
    def calculate_movement_magnitude(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['accel_magnitude'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        df['gyro_magnitude'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
        df['mag_magnitude'] = np.sqrt(df['mx']**2 + df['my']**2 + df['mz']**2)
        return df
    
    def apply_filters(self, df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
        df = df.copy()
        
        accel_cols = ['ax', 'ay', 'az']
        gyro_cols = ['gx', 'gy', 'gz']
        mag_cols = ['mx', 'my', 'mz']
        
        for col in accel_cols + gyro_cols + mag_cols:
            df[f'{col}_filtered'] = df[col].rolling(window=window_size, center=True).mean()
        
        df['accel_magnitude_filtered'] = np.sqrt(
            df['ax_filtered']**2 + df['ay_filtered']**2 + df['az_filtered']**2
        )
        
        return df
    
    def create_epochs(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['epoch'] = (df['TimeStamp'] // self.epoch_length).astype(int)
        
        epoch_stats = []
        
        for epoch_id in df['epoch'].unique():
            epoch_data = df[df['epoch'] == epoch_id]
            
            if len(epoch_data) < self.samples_per_epoch * 0.5:
                continue
            
            stats = {
                'epoch': epoch_id,
                'start_time': epoch_data['TimeStamp'].min(),
                'end_time': epoch_data['TimeStamp'].max(),
                'n_samples': len(epoch_data),
                'mean_ax': epoch_data['ax'].mean(),
                'mean_ay': epoch_data['ay'].mean(),
                'mean_az': epoch_data['az'].mean(),
                'std_ax': epoch_data['ax'].std(),
                'std_ay': epoch_data['ay'].std(),
                'std_az': epoch_data['az'].std(),
                'mean_accel_mag': epoch_data['accel_magnitude'].mean(),
                'std_accel_mag': epoch_data['accel_magnitude'].std(),
                'max_accel_mag': epoch_data['accel_magnitude'].max(),
                'min_accel_mag': epoch_data['accel_magnitude'].min(),
                'mean_gyro_mag': epoch_data['gyro_magnitude'].mean(),
                'std_gyro_mag': epoch_data['gyro_magnitude'].std(),
                'activity_counts': self.calculate_activity_counts(epoch_data),
                'movement_variance': np.var(epoch_data['accel_magnitude']),
                'zero_crossings': self.count_zero_crossings(epoch_data['ax'], epoch_data['ay'], epoch_data['az'])
            }
            
            epoch_stats.append(stats)
        
        return pd.DataFrame(epoch_stats)
    
    def calculate_activity_counts(self, epoch_data: pd.DataFrame) -> float:
        accel_filtered = np.sqrt(
            epoch_data['ax']**2 + epoch_data['ay']**2 + epoch_data['az']**2
        )
        
        gravity = 1000.0
        accel_filtered = accel_filtered - gravity
        
        activity_counts = np.sum(np.abs(accel_filtered))
        return activity_counts
    
    def count_zero_crossings(self, x: pd.Series, y: pd.Series, z: pd.Series) -> int:
        x_diff = np.diff(np.sign(x - x.mean()))
        y_diff = np.diff(np.sign(y - y.mean()))
        z_diff = np.diff(np.sign(z - z.mean()))
        
        zero_crossings = np.sum(x_diff != 0) + np.sum(y_diff != 0) + np.sum(z_diff != 0)
        return zero_crossings
    
    def detect_non_wear(self, df: pd.DataFrame, threshold: float = 50.0, 
                       min_duration: int = 60) -> pd.DataFrame:
        df = df.copy()
        
        low_activity = df['accel_magnitude'] < threshold
        
        df['non_wear'] = False
        current_bout_start = None
        
        for i, is_low in enumerate(low_activity):
            if is_low and current_bout_start is None:
                current_bout_start = i
            elif not is_low and current_bout_start is not None:
                bout_duration = (df.iloc[i]['TimeStamp'] - df.iloc[current_bout_start]['TimeStamp']) / 60
                if bout_duration >= min_duration:
                    df.loc[current_bout_start:i-1, 'non_wear'] = True
                current_bout_start = None
        
        return df
    
    def synchronize_datasets(self, orientation_df: pd.DataFrame, eeg_df: pd.DataFrame, 
                           tolerance: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        orient_times = orientation_df['TimeStamp'].values
        eeg_times = eeg_df['TimeStamp'].values
        
        common_start = max(orient_times.min(), eeg_times.min())
        common_end = min(orient_times.max(), eeg_times.max())
        
        orientation_sync = orientation_df[
            (orientation_df['TimeStamp'] >= common_start) & 
            (orientation_df['TimeStamp'] <= common_end)
        ].copy()
        
        eeg_sync = eeg_df[
            (eeg_df['TimeStamp'] >= common_start) & 
            (eeg_df['TimeStamp'] <= common_end)
        ].copy()
        
        return orientation_sync, eeg_sync
    
    def resample_to_common_rate(self, df: pd.DataFrame, target_rate: float) -> pd.DataFrame:
        df = df.copy()
        
        start_time = df['TimeStamp'].min()
        end_time = df['TimeStamp'].max()
        duration = end_time - start_time
        
        n_samples = int(duration * target_rate)
        new_timestamps = np.linspace(start_time, end_time, n_samples)
        
        resampled_data = {'TimeStamp': new_timestamps}
        
        for col in df.columns:
            if col != 'TimeStamp':
                resampled_data[col] = np.interp(new_timestamps, df['TimeStamp'], df[col])
        
        return pd.DataFrame(resampled_data)
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        quality_metrics = {
            'total_samples': len(df),
            'duration_hours': (df['TimeStamp'].max() - df['TimeStamp'].min()) / 3600,
            'sampling_rate_actual': len(df) / (df['TimeStamp'].max() - df['TimeStamp'].min()),
            'missing_samples': df.isnull().sum().sum(),
            'mean_accel_magnitude': df['accel_magnitude'].mean() if 'accel_magnitude' in df.columns else None,
            'std_accel_magnitude': df['accel_magnitude'].std() if 'accel_magnitude' in df.columns else None,
            'non_wear_percentage': (df['non_wear'].sum() / len(df)) * 100 if 'non_wear' in df.columns else None,
            'data_gaps': self.detect_data_gaps(df)
        }
        
        return quality_metrics
    
    def detect_data_gaps(self, df: pd.DataFrame, expected_interval: float = None) -> int:
        if expected_interval is None:
            expected_interval = 1.0 / self.sampling_rate
        
        time_diffs = np.diff(df['TimeStamp'])
        gap_threshold = expected_interval * 2
        gaps = np.sum(time_diffs > gap_threshold)
        
        return gaps
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str, 
                          include_metadata: bool = True) -> str:
        if include_metadata:
            quality_metrics = self.validate_data_quality(df)
            
            metadata_path = output_path.replace('.csv', '_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(quality_metrics, f, indent=2, default=str)
        
        df.to_csv(output_path, index=False)
        return output_path
    
    def process_orientation_pipeline(self, orientation_path: str, output_dir: str, 
                                   eeg_path: Optional[str] = None) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        
        orientation_df = self.load_orientation_data(orientation_path)
        
        orientation_df = self.calculate_movement_magnitude(orientation_df)
        orientation_df = self.apply_filters(orientation_df)
        orientation_df = self.detect_non_wear(orientation_df)
        
        if eeg_path:
            eeg_df = self.load_eeg_data(eeg_path)
            orientation_df, eeg_df = self.synchronize_datasets(orientation_df, eeg_df)
            
            eeg_output_path = os.path.join(output_dir, 'eeg_synchronized.csv')
            eeg_df.to_csv(eeg_output_path, index=False)
        
        accel_csv_path = os.path.join(output_dir, 'accelerometer_data.csv')
        self.create_accelerometer_csv(orientation_df, accel_csv_path)
        
        epochs_df = self.create_epochs(orientation_df)
        epochs_path = os.path.join(output_dir, 'movement_epochs.csv')
        epochs_df.to_csv(epochs_path, index=False)
        
        processed_path = os.path.join(output_dir, 'orientation_processed.csv')
        self.save_processed_data(orientation_df, processed_path)
        
        output_files = {
            'accelerometer_csv': accel_csv_path,
            'movement_epochs': epochs_path,
            'processed_orientation': processed_path
        }
        
        if eeg_path:
            output_files['synchronized_eeg'] = eeg_output_path
        
        return output_files

def main():
    preprocessor = OrientationPreprocessor(sampling_rate=25.0)
    
    orientation_file = "data/Mentalab-sleep-analysis_ORN.csv"
    eeg_file = "data/Mentalab-sleep-analysis_ExG.csv"
    output_directory = "processed_data"
    
    try:
        output_files = preprocessor.process_orientation_pipeline(
            orientation_path=orientation_file,
            output_dir=output_directory,
            eeg_path=eeg_file
        )
        
        print("Processing completed successfully!")
        print("Output files:")
        for key, path in output_files.items():
            print(f"  {key}: {path}")
            
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()