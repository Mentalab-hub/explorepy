import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Load and process data
sleep_data = pd.read_csv("sleep_staging_results/hypnogram.csv")
activity_data = pd.read_csv("activity_data_results/categorized_activity_data.csv")

activity_data['timestamp'] = pd.to_datetime(activity_data['timestamp'])
activity_start = activity_data['timestamp'].min()
sleep_data['datetime'] = activity_start + pd.to_timedelta(sleep_data['TimeStamp'], unit='s')

sleep_end_time = sleep_data['datetime'].max()
activity_data = activity_data[activity_data['timestamp'] <= sleep_end_time].copy()

activity_data['hours_from_start'] = (activity_data['timestamp'] - activity_start).dt.total_seconds() / 3600
sleep_data['hours_from_start'] = (sleep_data['datetime'] - activity_start).dt.total_seconds() / 3600

# Color palettes
sleep_colors = {
    'Wake': '#E74C3C',        # Bold red
    'Light Sleep': '#3498DB',  # Bright blue
    'Deep Sleep': '#2C3E50',   # Dark navy
    'REM Sleep': '#27AE60'     # Green
}

activity_colors = {
    'Sedentary': '#95A5A6',     # Gray
    'Light': '#F39C12',         # Orange
    'Moderate': '#E67E22',      # Dark orange
    'Vigorous': '#D35400',      # Red-orange
    'Very Vigorous': '#8E44AD'  # Purple
}

sleep_stage_map = {
    'W': 'Wake',
    'N1': 'Light Sleep',
    'N2': 'Deep Sleep', 
    'N3': 'Deep Sleep',
    'REM': 'REM Sleep'
}

sleep_data['sleep_stage_name'] = sleep_data['sleep_stage'].map(sleep_stage_map)

# 1. ACTIVITY OVER TIME
def plot_activity_over_time():
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot activity with color coding
    for activity_level in sorted(activity_data['activity_level'].unique()):
        level_data = activity_data[activity_data['activity_level'] == activity_level]
        ax.scatter(level_data['hours_from_start'], level_data['acceleration'],
                  c=activity_colors.get(activity_level, '#808080'),
                  label=activity_level, alpha=0.7, s=15)
    
    # Add trend line
    activity_smooth = activity_data.set_index('hours_from_start')['acceleration'].rolling(window=30, center=True).mean()
    ax.plot(activity_smooth.index, activity_smooth.values, 
           color='black', linewidth=2, alpha=0.8, label='Rolling Average')
    
    ax.set_xlabel('Hours from Start', fontsize=12)
    ax.set_ylabel('Acceleration', fontsize=12)
    ax.set_title('Activity Levels Over Time', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show hours with better spacing
    max_hours = activity_data['hours_from_start'].max()
    ax.set_xlim(0, max_hours)
    hour_ticks = np.arange(0, max_hours + 1, 1)  # Every hour
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels([f'{h:.0f}h' for h in hour_ticks])
    
    plt.tight_layout()
    plt.savefig('1_activity_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()

# 2. SLEEP STAGES OVER TIME (Hypnogram)
def plot_sleep_stages_over_time():
    fig, ax = plt.subplots(figsize=(16, 6))
    
    stage_numeric = {'Deep Sleep': 1, 'Light Sleep': 2, 'REM Sleep': 3, 'Wake': 4}
    sleep_data['stage_numeric'] = sleep_data['sleep_stage_name'].map(stage_numeric)
    
    time_points = sleep_data['hours_from_start'].values
    stage_values = sleep_data['stage_numeric'].values
    
    ax.step(time_points, stage_values, where='post', linewidth=2, color='black', alpha=0.8)
    
    for stage, numeric_val in stage_numeric.items():
        mask = sleep_data['stage_numeric'] == numeric_val
        if mask.any():
            ax.fill_between(time_points, 0, stage_values, 
                          where=(sleep_data['stage_numeric'] == numeric_val),
                          color=sleep_colors[stage], alpha=0.7, step='post', label=stage)
    
    ax.set_xlabel('Hours from Start', fontsize=12)
    ax.set_ylabel('Sleep Stage', fontsize=12)
    ax.set_title('Sleep Stages Over Time (Hypnogram)', fontsize=16, fontweight='bold')
    ax.set_yticks(list(stage_numeric.values()))
    ax.set_yticklabels(list(stage_numeric.keys()))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 4.5)
    
    max_hours = sleep_data['hours_from_start'].max()
    ax.set_xlim(0, max_hours)
    hour_ticks = np.arange(0, max_hours + 1, 1)  # Every hour
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels([f'{h:.0f}h' for h in hour_ticks])
    
    plt.tight_layout()
    plt.savefig('2_sleep_stages_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()

# 3. OVERLAYED BOTH
def plot_overlayed():
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    for activity_level in sorted(activity_data['activity_level'].unique()):
        level_data = activity_data[activity_data['activity_level'] == activity_level]
        ax1.scatter(level_data['hours_from_start'], level_data['acceleration'],
                   c=activity_colors.get(activity_level, '#808080'),
                   label=f'Activity: {activity_level}', alpha=0.6, s=12)
    
    activity_smooth = activity_data.set_index('hours_from_start')['acceleration'].rolling(window=30, center=True).mean()
    ax1.plot(activity_smooth.index, activity_smooth.values, 
            color='darkblue', linewidth=3, alpha=0.9, label='Activity Trend')
    
    ax1.set_xlabel('Hours from Start', fontsize=12)
    ax1.set_ylabel('Acceleration', fontsize=12, color='darkblue')
    ax1.tick_params(axis='y', labelcolor='darkblue')
    
    ax2 = ax1.twinx()
    
    stage_numeric = {'Deep Sleep': 1, 'Light Sleep': 2, 'REM Sleep': 3, 'Wake': 4}
    sleep_data['stage_numeric'] = sleep_data['sleep_stage_name'].map(stage_numeric)
    
    time_points = sleep_data['hours_from_start'].values
    stage_values = sleep_data['stage_numeric'].values
    
    ax2.step(time_points, stage_values, where='post', linewidth=3, 
            color='darkred', alpha=0.8, label='Sleep Stages')
    
    for stage, numeric_val in stage_numeric.items():
        mask = sleep_data['stage_numeric'] == numeric_val
        if mask.any():
            ax2.fill_between(time_points, 0, stage_values, 
                           where=(sleep_data['stage_numeric'] == numeric_val),
                           color=sleep_colors[stage], alpha=0.4, step='post', 
                           label=f'Sleep: {stage}')
    
    ax2.set_ylabel('Sleep Stage', fontsize=12, color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.set_yticks(list(stage_numeric.values()))
    ax2.set_yticklabels(list(stage_numeric.keys()))
    ax2.set_ylim(0.5, 4.5)
    
    max_hours = max(activity_data['hours_from_start'].max(), sleep_data['hours_from_start'].max())
    ax1.set_xlim(0, max_hours)
    ax2.set_xlim(0, max_hours)
    hour_ticks = np.arange(0, max_hours + 1, 1)
    ax1.set_xticks(hour_ticks)
    ax1.set_xticklabels([f'{h:.0f}h' for h in hour_ticks])
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
    
    plt.title('Sleep Stages and Activity Data Overlay', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('3_overlayed_sleep_activity.png', dpi=300, bbox_inches='tight')
    plt.show()

print("Creating 3 essential diagrams...")

print("\n1. Activity Over Time")
plot_activity_over_time()

print("\n2. Sleep Stages Over Time") 
plot_sleep_stages_over_time()

print("\n3. Overlayed Sleep & Activity")
plot_overlayed()

print("\nCompleted! Generated files:")
print("- 1_activity_over_time.png")
print("- 2_sleep_stages_over_time.png") 
print("- 3_overlayed_sleep_activity.png")

print(f"\nData Summary:")
print(f"Activity data points: {len(activity_data):,}")
print(f"Sleep data points: {len(sleep_data):,}")
print(f"Duration: {activity_data['hours_from_start'].max():.1f} hours")

sleep_summary = sleep_data['sleep_stage_name'].value_counts()
print(f"\nSleep stage distribution:")
for stage, count in sleep_summary.items():
    percentage = (count / len(sleep_data)) * 100
    print(f"  {stage}: {percentage:.1f}%")

activity_summary = activity_data['activity_level'].value_counts()
print(f"\nActivity level distribution:")
for level, count in activity_summary.items():
    percentage = (count / len(activity_data)) * 100
    print(f"  {level}: {percentage:.1f}%")