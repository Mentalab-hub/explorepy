import pandas as pd
import json
from pyActigraphy.io.base import BaseRaw
from datetime import datetime
import uuid as uuid_lib
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("accelerometer_results/processed_acc_data.csv.gz", compression='gzip')

df['time'] = df['time'].str.split(r' \[').str[0]
df['time'] = df['time'].str.replace(r"\+0000", "+00:00", regex=True)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
df.index.freq = "30S"

with open("accelerometer_results/processed_acc_data-summary.json", "r") as f:
    meta = json.load(f)

name = meta["file-name"]
uuid_val = uuid_lib.uuid4()
fmt = "BBA"
axial_mode = "uni"
raw_time_str = meta["file-startTime"]
cleaned_time_str = raw_time_str.split(' [')[0]

period = df.index.max() - df.index.min() + pd.Timedelta(seconds=30)
frequency = 1 / period.total_seconds()

df = df.reset_index()

df['time'] = pd.to_datetime(df['time']).dt.floor('30S')

full_range = pd.date_range(start=df['time'].min().floor('30S'), 
                           end=df['time'].max().ceil('30S'), 
                           freq='30S', 
                           tz='UTC')

df = df.set_index('time').reindex(full_range).interpolate(method='time')

start_time = df.index.min()
data = df["acc"]
light = None

act = BaseRaw(
    name=name,
    uuid=uuid_val,
    format=fmt,
    axial_mode=axial_mode,
    start_time=start_time,
    data=data,
    light=light,
    fpath="accelerometer_results/acc_data.csv.gz",
    period=period,
    frequency=frequency
)

cut_points = [8, 20, 60, 120]
labels = ["Sedentary", "Light", "Moderate", "Vigorous", "Very Vigorous"]

act.create_activity_report(
    cut_points=cut_points,
    labels=labels,
    threshold=10,
    start_time="06:00:00",
    stop_time="22:00:00",
    oformat="minute",
    verbose=True
)

report = act.activity_report
print("Activity Report:")
print(report)

report.to_csv("activity_report.csv", index=False)
print("\nActivity report saved to 'activity_report.csv'")

processed_data = pd.DataFrame({
    'timestamp': act.data.index,
    'acceleration': act.data.values
})
processed_data.to_csv("processed_accelerometer_data.csv", index=False)
print("Processed data saved to 'processed_accelerometer_data.csv'")

def categorize_activity(value, cut_points, labels):
    """Categorize activity level based on cut points"""
    if pd.isna(value):
        return "Unknown"
    for i, cp in enumerate(cut_points):
        if value <= cp:
            return labels[i]
    return labels[-1]

activity_data = processed_data.copy()
activity_data['activity_level'] = activity_data['acceleration'].apply(
    lambda x: categorize_activity(x, cut_points, labels)
)
activity_data['hour'] = activity_data['timestamp'].dt.hour
activity_data['date'] = activity_data['timestamp'].dt.date

activity_data.to_csv("categorized_activity_data.csv", index=False)
print("Categorized data saved to 'categorized_activity_data.csv'")

plt.figure(figsize=(14, 8))
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']

for level, color in zip(labels, colors):
    level_data = activity_data[activity_data['activity_level'] == level]
    if not level_data.empty:
        plt.scatter(level_data['timestamp'], level_data['acceleration'], 
                   c=color, label=level, alpha=0.6, s=1)

plt.title('Activity Pattern Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.legend(title='Activity Level')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('activity_pattern_over_time.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print(f"Total data points: {len(activity_data):,}")
print(f"Time range: {activity_data['timestamp'].min()} to {activity_data['timestamp'].max()}")
print(f"Duration: {activity_data['timestamp'].max() - activity_data['timestamp'].min()}")
print(f"\nAcceleration statistics:")
print(f"  Mean: {activity_data['acceleration'].mean():.2f}")
print(f"  Median: {activity_data['acceleration'].median():.2f}")
print(f"  Std: {activity_data['acceleration'].std():.2f}")
print(f"  Min: {activity_data['acceleration'].min():.2f}")
print(f"  Max: {activity_data['acceleration'].max():.2f}")

print(f"\nActivity level breakdown:")
for level in labels:
    count = (activity_data['activity_level'] == level).sum()
    percentage = (count / len(activity_data)) * 100
    print(f"  {level}: {count:,} ({percentage:.1f}%)")

print(f"\nFiles saved:")
print(f"  - activity_report.csv")
print(f"  - processed_accelerometer_data.csv") 
print(f"  - categorized_activity_data.csv")
print(f"  - activity_pattern_over_time.png")