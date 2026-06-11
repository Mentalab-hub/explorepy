import pandas as pd
import numpy as np
import mne
import yasa
import os
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

class SleepStager:
    def __init__(self, sampling_rate: float = 250.0):
        self.sampling_rate = sampling_rate
        self.target_sf = 100.0
        self.epoch_length = 30.0
        
    def load_eeg_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        df['TimeStamp'] = pd.to_numeric(df['TimeStamp'], errors='coerce')
        df = df.dropna(subset=['TimeStamp'])
        df = df.sort_values('TimeStamp').reset_index(drop=True)
        return df
    
    def create_mne_raw(self, eeg_df: pd.DataFrame) -> mne.io.RawArray:
        ch_names = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
        ch_types = ['eeg'] * 8
        
        eeg_channels = [col for col in eeg_df.columns if col.startswith('ch')]
        if len(eeg_channels) != 8:
            raise ValueError(f"Expected 8 channels, found {len(eeg_channels)}")
        
        data = eeg_df[eeg_channels].values.T
        
        info = mne.create_info(ch_names=ch_names, sfreq=self.sampling_rate, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)
        
        raw.apply_function(lambda x: x * 1e-6)
        
        return raw
    
    def preprocess_eeg(self, raw: mne.io.RawArray) -> mne.io.RawArray:
        raw_copy = raw.copy()
        raw_copy.filter(0.1, 40)
        
        if raw_copy.info['sfreq'] != 100:
            raw_copy.resample(100)
        
        return raw_copy
    
    def perform_sleep_staging(self, raw: mne.io.RawArray, 
                            eeg_channel: str = 'ch1') -> Dict[str, Any]:
        
        if eeg_channel not in raw.ch_names:
            eeg_channel = raw.ch_names[0]
            print(f"Channel {eeg_channel} not found, using {raw.ch_names[0]}")
        
        print(f"Using YASA with channel: {eeg_channel}")
        print(f"Data shape: {raw.get_data().shape}")
        print(f"Sampling rate: {raw.info['sfreq']} Hz")
        print(f"Duration: {raw.times[-1]:.1f} seconds")
        
        sls = yasa.SleepStaging(raw, eeg_name=eeg_channel)
        
        y_pred = sls.predict()
        y_proba = sls.predict_proba()
        
        confidence = np.max(y_proba, axis=1)
        
        hypno = yasa.Hypnogram(y_pred)
        hypno_int = yasa.hypno_str_to_int(y_pred)
        hypno_up = yasa.hypno_upsample_to_data(
            hypno=hypno_int, 
            sf_hypno=(1/30), 
            data=raw.get_data(), 
            sf_data=raw.info['sfreq']
        )
        
        results = {
            'hypnogram': y_pred,
            'hypnogram_int': hypno_int,
            'hypnogram_upsampled': hypno_up,
            'hypno_object': hypno,
            'probabilities': y_proba,
            'confidence': confidence,
            'staging_object': sls
        }
        
        print(f"Sleep staging completed: {len(y_pred)} epochs")
        unique, counts = np.unique(y_pred, return_counts=True)
        for stage, count in zip(unique, counts):
            print(f"  {stage}: {count} epochs ({count/len(y_pred)*100:.1f}%)")
        
        return results
    
    def create_hypnogram_dataframe(self, results: Dict[str, Any], 
                                 start_timestamp: float) -> pd.DataFrame:
        hypnogram = results['hypnogram']
        confidence = results['confidence']
        
        n_epochs = len(hypnogram)
        timestamps = []
        
        for i in range(n_epochs):
            epoch_timestamp = start_timestamp + (i * self.epoch_length)
            timestamps.append(epoch_timestamp)
        
        hypno_df = pd.DataFrame({
            'TimeStamp': timestamps,
            'sleep_stage': hypnogram,
            'sleep_stage_int': results['hypnogram_int'],
            'confidence': confidence,
            'epoch': range(n_epochs)
        })
        
        return hypno_df
    
    def calculate_sleep_statistics(self, hypnogram: np.ndarray) -> Dict[str, float]:
        stage_counts = pd.Series(hypnogram).value_counts()
        total_epochs = len(hypnogram)
        
        stats = {
            'total_epochs': total_epochs,
            'total_recording_time_min': total_epochs * 0.5,
            'total_sleep_time_min': (total_epochs - stage_counts.get('W', 0)) * 0.5,
            'sleep_efficiency': ((total_epochs - stage_counts.get('W', 0)) / total_epochs) * 100,
            'wake_pct': (stage_counts.get('W', 0) / total_epochs) * 100,
            'n1_pct': (stage_counts.get('N1', 0) / total_epochs) * 100,
            'n2_pct': (stage_counts.get('N2', 0) / total_epochs) * 100,
            'n3_pct': (stage_counts.get('N3', 0) / total_epochs) * 100,
            'rem_pct': (stage_counts.get('R', 0) / total_epochs) * 100,
            'sleep_onset_latency_min': self.find_sleep_onset(hypnogram),
            'rem_latency_min': self.find_rem_onset(hypnogram),
            'waso_min': self.calculate_waso(hypnogram),
            'number_of_awakenings': self.count_awakenings(hypnogram)
        }
        
        return stats
    
    def find_sleep_onset(self, hypnogram: np.ndarray) -> float:
        for i, stage in enumerate(hypnogram):
            if stage != 'W':
                return i * 0.5
        return 0.0
    
    def find_rem_onset(self, hypnogram: np.ndarray) -> float:
        sleep_start = None
        for i, stage in enumerate(hypnogram):
            if stage != 'W' and sleep_start is None:
                sleep_start = i
            if stage == 'R' and sleep_start is not None:
                return (i - sleep_start) * 0.5
        return 0.0
    
    def calculate_waso(self, hypnogram: np.ndarray) -> float:
        sleep_start = None
        wake_epochs = 0
        
        for i, stage in enumerate(hypnogram):
            if stage != 'W' and sleep_start is None:
                sleep_start = i
            elif stage == 'W' and sleep_start is not None:
                wake_epochs += 1
        
        return wake_epochs * 0.5
    
    def count_awakenings(self, hypnogram: np.ndarray) -> int:
        awakenings = 0
        in_sleep = False
        
        for stage in hypnogram:
            if stage != 'W' and not in_sleep:
                in_sleep = True
            elif stage == 'W' and in_sleep:
                awakenings += 1
                in_sleep = False
        
        return awakenings
    
    def create_stage_transitions(self, hypnogram: np.ndarray) -> pd.DataFrame:
        transitions = []
        
        for i in range(1, len(hypnogram)):
            if hypnogram[i] != hypnogram[i-1]:
                transitions.append({
                    'epoch': i,
                    'from_stage': hypnogram[i-1],
                    'to_stage': hypnogram[i],
                    'timestamp_min': i * 0.5
                })
        
        return pd.DataFrame(transitions)
    
    def analyze_bandpower(self, raw: mne.io.RawArray, 
                         hypno_up: np.ndarray, 
                         eeg_channel: str = 'ch1') -> pd.DataFrame:
        
        data_filt = raw.get_data() * 1e6
        ch_idx = raw.ch_names.index(eeg_channel)
        data_channel = data_filt[ch_idx]
        
        bandpower_stages = yasa.bandpower(
            data_channel, 
            sf=raw.info['sfreq'], 
            win_sec=4, 
            relative=True, 
            hypno=hypno_up, 
            include=(0, 1, 2, 3, 4)
        )
        
        bandpower_avg = bandpower_stages.groupby('Stage')[['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma']].mean()
        bandpower_avg.index = ['Wake', 'N1', 'N2', 'N3', 'REM']
        
        return bandpower_avg
    
    def detect_sleep_events(self, raw: mne.io.RawArray, 
                           hypno_up: np.ndarray, 
                           eeg_channel: str = 'ch1') -> Dict[str, Any]:
        
        data_filt = raw.get_data() * 1e6
        ch_idx = raw.ch_names.index(eeg_channel)
        data_channel = data_filt[ch_idx]
        
        sw = yasa.sw_detect(data_channel, raw.info['sfreq'], hypno=hypno_up)
        sp = yasa.spindles_detect(data_channel, raw.info['sfreq'])
        
        return {
            'slow_waves': sw,
            'spindles': sp,
            'sw_summary': sw.summary() if sw is not None else None,
            'sp_summary': sp.summary() if sp is not None else None
        }
    
    def save_results(self, hypno_df: pd.DataFrame, 
                    sleep_stats: Dict[str, float],
                    transitions_df: pd.DataFrame,
                    bandpower_df: pd.DataFrame,
                    output_dir: str) -> Dict[str, str]:
        
        os.makedirs(output_dir, exist_ok=True)
        
        hypno_path = os.path.join(output_dir, 'hypnogram.csv')
        hypno_df.to_csv(hypno_path, index=False)
        
        stats_path = os.path.join(output_dir, 'sleep_statistics.csv')
        stats_df = pd.DataFrame([sleep_stats])
        stats_df.to_csv(stats_path, index=False)
        
        transitions_path = os.path.join(output_dir, 'stage_transitions.csv')
        transitions_df.to_csv(transitions_path, index=False)
        
        bandpower_path = os.path.join(output_dir, 'bandpower_by_stage.csv')
        bandpower_df.to_csv(bandpower_path, index=True)
        
        return {
            'hypnogram': hypno_path,
            'statistics': stats_path,
            'transitions': transitions_path,
            'bandpower': bandpower_path
        }
    
    def process_sleep_staging_pipeline(self, eeg_filepath: str, 
                                     output_dir: str,
                                     eeg_channel: str = 'ch1') -> Dict[str, Any]:
        
        print("Loading EEG data...")
        eeg_df = self.load_eeg_data(eeg_filepath)
        
        print("Creating MNE Raw object...")
        raw = self.create_mne_raw(eeg_df)
        
        print("Preprocessing EEG...")
        raw_clean = self.preprocess_eeg(raw)
        
        print("Performing sleep staging...")
        staging_results = self.perform_sleep_staging(raw_clean, eeg_channel=eeg_channel)
        
        print("Creating hypnogram dataframe...")
        start_timestamp = eeg_df['TimeStamp'].iloc[0]
        hypno_df = self.create_hypnogram_dataframe(staging_results, start_timestamp)
        
        print("Calculating sleep statistics...")
        sleep_stats = self.calculate_sleep_statistics(staging_results['hypnogram'])
        
        print("Analyzing stage transitions...")
        transitions_df = self.create_stage_transitions(staging_results['hypnogram'])
        
        print("Analyzing bandpower by sleep stage...")
        bandpower_df = self.analyze_bandpower(raw_clean, staging_results['hypnogram_upsampled'], eeg_channel)
        
        print("Detecting sleep events...")
        sleep_events = self.detect_sleep_events(raw_clean, staging_results['hypnogram_upsampled'], eeg_channel)
        
        print("Saving results...")
        output_files = self.save_results(hypno_df, sleep_stats, transitions_df, bandpower_df, output_dir)
        
        results = {
            'hypnogram_df': hypno_df,
            'sleep_statistics': sleep_stats,
            'transitions_df': transitions_df,
            'bandpower_df': bandpower_df,
            'sleep_events': sleep_events,
            'staging_results': staging_results,
            'output_files': output_files,
            'raw_data': raw_clean
        }
        
        print(f"\nSleep staging completed!")
        print(f"Total recording time: {sleep_stats['total_recording_time_min']:.1f} minutes")
        print(f"Total sleep time: {sleep_stats['total_sleep_time_min']:.1f} minutes")
        print(f"Sleep efficiency: {sleep_stats['sleep_efficiency']:.1f}%")
        print(f"Sleep onset latency: {sleep_stats['sleep_onset_latency_min']:.1f} minutes")
        print(f"REM latency: {sleep_stats['rem_latency_min']:.1f} minutes")
        print(f"Wake: {sleep_stats['wake_pct']:.1f}%")
        print(f"N1: {sleep_stats['n1_pct']:.1f}%")
        print(f"N2: {sleep_stats['n2_pct']:.1f}%")
        print(f"N3: {sleep_stats['n3_pct']:.1f}%")
        print(f"REM: {sleep_stats['rem_pct']:.1f}%")
        
        return results

def main():
    stager = SleepStager(sampling_rate=250.0)
    
    eeg_file = "processed_data/eeg_synchronized.csv"
    output_directory = "sleep_staging_results"
    
    results = stager.process_sleep_staging_pipeline(
        eeg_filepath=eeg_file,
        output_dir=output_directory,
        eeg_channel='ch1'
    )
    
    print("\nOutput files created:")
    for key, path in results['output_files'].items():
        print(f"  {key}: {path}")
    
    print(f"\nBandpower by sleep stage:")
    print(results['bandpower_df'])
    
    if results['sleep_events']['sw_summary'] is not None:
        print(f"\nSlow waves detected: {len(results['sleep_events']['sw_summary'])} events")
    
    if results['sleep_events']['sp_summary'] is not None:
        print(f"Sleep spindles detected: {len(results['sleep_events']['sp_summary'])} events")

if __name__ == "__main__":
    main()