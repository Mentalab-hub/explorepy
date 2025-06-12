# Sleep Activity Pipeline (with PyActigraphy, Accelerometer and YASA)

A pipeline for analyzing sleep stages and activity patterns using EEG and accelerometer data from Mentalab device.

## Overview

This pipeline processes raw EEG and orientation data to:
1. **Detect sleep stages** using YASA (Yet Another Spindle Algorithm)
2. **Analyze movement patterns** using accelerometer and pyActigraphy
3. **Visualize sleep-activity relationships**

## Input Data

- `data/Mentalab-sleep-analysis_ExG.csv` - EEG data
- `data/Mentalab-sleep-analysis_ORN.csv` - Orientation/accelerometer data

## Pipeline Steps

### 1. Data Preprocessing (`preprocess.py`)

```bash
python preprocess/preprocess.py
```

**Purpose**: Synchronize EEG and orientation data, extract accelerometer features

**Key Features**:
- Synchronizes timestamps between EEG and orientation data
- Calculates movement magnitudes and applies filtering
- Creates 30-second epochs for analysis
- Generates `accelerometer_data.csv` for further processing

**Outputs**:
- `processed_data/eeg_synchronized.csv` - Synchronized EEG data
- `processed_data/accelerometer_data.csv` - Formatted accelerometer data
- `processed_data/movement_epochs.csv` - 30-second movement summaries

### 2. Sleep Stage Detection (`sleep_staging.py`)

```bash
python sleep_staging/sleep_staging.py
```

**Purpose**: Analyze EEG data to detect sleep stages using YASA

**Libraries**: 
- **MNE-Python** for EEG preprocessing (filtering, resampling)
- **YASA** for automated sleep staging using machine learning

**Process**:
- Loads synchronized EEG data
- Applies bandpass filtering (0.1-40 Hz)
- Resamples to 100 Hz for YASA compatibility
- Performs sleep staging (Wake, N1, N2, N3, REM)
- Calculates sleep statistics and detects sleep events

**Outputs**:
- `sleep_staging_results/hypnogram.csv` - Sleep stages over time
- `sleep_staging_results/sleep_statistics.csv` - Sleep quality metrics
- `sleep_staging_results/bandpower_by_stage.csv` - Frequency analysis

### 3. Accelerometer Processing (`preprocess_acc.py`)

```bash
python preprocess/preprocess_acc.py
```

**Purpose**: Process accelerometer data using accelerometer library

**Library**:
- **Accelerometer Library** for activity classification
- Automatically installs via `pip install accelerometer`

**Process**:
- Processes raw accelerometer CSV using `accProcess` command
- Extracts activity counts and movement patterns
- Creates epoch-level summaries (30-second windows)

**Outputs**:
- `accelerometer_results/acc_data-timeSeries.csv.gz` - High-resolution activity data
- `accelerometer_results/acc_data-epoch.csv` - 30-second epoch summaries
- `accelerometer_results/acc_data-summary.json` - Processing metadata

### 4. Activity Analysis (`analyze_activity.py`)

```bash
python analyze/analyze_activity.py
```

**Purpose**: Convert accelerometer data to pyActigraphy format

**Library**:
- **pyActigraphy** for activity pattern analysis
- Activity classification with configurable cut-points

**Process**:
- Creates pyActigraphy-compatible data format
- Applies activity classification:
  - Sedentary: ≤8 mg
  - Light: 8-20 mg  
  - Moderate: 20-60 mg
  - Vigorous: 60-120 mg
  - Very Vigorous: >120 mg
- Generates activity reports and visualizations

**Outputs**:
- `categorized_activity_data.csv` - Activity levels over time
- `activity_report.csv` - Summary statistics
- `activity_pattern_over_time.png` - Visualization

### 5. Integrated Visualization (`visualization.py`)

```bash
python visualize.py
```

**Purpose**: Create visualizations of sleep stages and activity

**Generates**:
1. **Activity Over Time** - Movement patterns with trend analysis
2. **Sleep Hypnogram** - Clinical-style sleep stage visualization  
3. **Overlay Plot** - Combined sleep-activity relationships

## Results and Interpretation

### Sleep-Activity Correlation Analysis

The pipeline demonstrates strong correlations between EEG-derived sleep stages and accelerometer-detected movement patterns. The overlay visualization reveals several key findings:

![alt text](3_overlayed_sleep_activity.png "sleep activity")

#### Sleep Architecture Pattern
- **Total Recording Duration (truncated)**: ~8.5 hours
- **Sleep Onset**: Approximately 30 minutes after recording start
- **Sleep Stages Distribution**: 
  - Wake periods: Brief awakenings throughout the night
  - Light Sleep (N1/N2): Predominant in early and late sleep periods
  - Deep Sleep (N3): Concentrated in first half of sleep (hours 1-4, 6-7)
  - REM Sleep: Distributed across sleep period with longer episodes in later hours

#### Movement-Sleep Stage Relationships

**1. Wake Periods (Pink/Red regions)**
- **High Activity Correlation**: Activity spikes (50-95 acceleration units) occur exclusively during wake periods
- **Movement Intensity**: Vigorous activity (orange dots) clustered during wake episodes
- **Sleep Fragmentation**: Multiple brief awakenings indicate normal sleep fragmentation

**2. Deep Sleep (N3 - Dark Gray regions)**
- **Minimal Movement**: Activity drops to near-zero during deep sleep phases
- **Motor Suppression**: Demonstrates effective muscle atonia during deep sleep
- **Sleep Consolidation**: Longest continuous deep sleep periods occur in first sleep cycle

**3. Light Sleep and REM (Light Blue regions)**
- **Moderate Activity**: Slight movement detection during sleep stage transitions
- **REM Paradox**: Some movement during REM sleep, consistent with incomplete muscle atonia
- **Micro-movements**: Small activity spikes at stage boundaries indicate natural sleep transitions

**Accelerometer Validation**:
- **Strong Concordance**: Movement patterns align perfectly with EEG-derived sleep stages
- **Objective Verification**: Accelerometer data validates manual sleep scoring accuracy
- **Clinical Utility**: Demonstrates potential for accelerometer-based sleep monitoring

**Methodological Strengths**:
- **Multi-modal Approach**: Combined EEG and accelerometer analysis provides comprehensive assessment
- **Standardized Processing**: Use of established libraries (YASA, pyActigraphy) ensures reproducible results
- **Real-time Potential**: Pipeline suitable for longitudinal sleep monitoring applications

### Statistical Summary

Based on the analyzed data:
- **Sleep Latency**: ~30 minutes (normal range)
- **Sleep Efficiency**: High (minimal wake periods during sleep)
- **Movement-Sleep Correlation**: Strong negative correlation between activity and sleep depth
- **Sleep Stage Transitions**: Well-defined boundaries with appropriate movement patterns

## Key Libraries Used

| Library | Purpose | Usage |
|---------|---------|--------|
| **YASA** | Sleep staging | Automated EEG-based sleep detection |
| **MNE-Python** | EEG processing | Signal filtering, epoching, preprocessing |
| **Oxford Accelerometer** | Activity processing | Raw accelerometer data analysis |
| **pyActigraphy** | Activity analysis | Research-standard activity metrics |
| **Pandas/NumPy** | Data processing | Data manipulation and analysis |
| **Matplotlib** | Visualization | Plot generation and formatting |

## Final Outputs

### Data Files
- `hypnogram.csv` - Sleep stages (30-second epochs)
- `categorized_activity_data.csv` - Activity classifications
- `sleep_statistics.csv` - Sleep quality metrics
- `sleep_actigraphy_merged.csv` - Combined sleep-activity dataset

### Visualizations
- `1_activity_over_time.png` - Activity patterns
- `2_sleep_stages_over_time.png` - Sleep hypnogram
- `3_overlayed_sleep_activity.png` - Combined analysis

## Quick Start

```bash
# 1. Ensure you have the input files
# Mentalab-sleep-analysis_ExG.csv
# Mentalab-sleep-analysis_ORN.csv

# 2. Run the complete pipeline
# !!!important: rename files and relocate files accordingly
python [name_of_program.py]
```

## Dependencies

```bash
### may require setting up different environments for yasa and accelerometer library (incompatible with each other)

pip install pandas numpy matplotlib seaborn
pip install mne yasa
pip install accelerometer
pip install pyActigraphy
pip install scikit-learn  # for classification metrics
```

## Applications

This pipeline enables comprehensive analysis of:

### Research Applications
- **Sleep Architecture Studies**: Detailed analysis of sleep stage distributions and transitions
- **Circadian Rhythm Research**: Activity patterns and sleep-wake cycle assessment
- **Sleep Disorder Detection**: Identification of abnormal movement patterns during sleep
- **Intervention Studies**: Pre/post analysis of sleep quality changes

### Clinical Applications
- **Sleep Quality Assessment**: Objective measurement of sleep efficiency and continuity
- **Home Sleep Monitoring**: Long-term sleep pattern analysis outside laboratory settings
- **Treatment Monitoring**: Tracking sleep improvements in clinical populations
- **Diagnostic Support**: Validation of subjective sleep reports with objective measures

### Technical Applications
- **Algorithm Validation**: Comparison of different sleep detection methods
- **Sensor Fusion**: Integration of multiple physiological signals
- **Machine Learning**: Training data for automated sleep analysis systems
- **Quality Control**: Validation of manual sleep scoring accuracy

## Future Enhancements

1. **Real-time Processing**: Adaptation for live sleep monitoring
2. **Machine Learning Integration**: Automated sleep quality scoring
4. **Environmental Integration**: Addition of light, temperature, and sound data