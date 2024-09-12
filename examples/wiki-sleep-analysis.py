# Data processing
## Load packages
import mne
import yasa
import numpy as np
import matplotlib.pyplot as plt

## Load data
ch_names = ['EOG-above', 'EOG-canthus', 'EMG-chin', 'Dry-O2', 'Dry-C2', 'Dry-F2', 'Sticky-C1', 'Wet-Fc1']
data = np.loadtxt("wiki-sleep-analysis-data/Mentalab-sleep-analysis_ExG.csv", skiprows=1, delimiter=',').transpose()[1:9]
sf = 250
ch_types = ["eog", "eog", "emg", "eeg", "eeg", "eeg", "eeg", "eeg"]
info = mne.create_info(ch_names = ch_names, sfreq = sf, ch_types = ch_types)
raw = mne.io.RawArray(data, info)
raw.apply_function(lambda x: x * 1e-6) # scale muV to V

## Pre-processing
raw.filter(0.1, 40) # Apply a bandpass filter from 0.1 to 40 Hz

# Detecting sleep stages
sls = yasa.SleepStaging(raw, eeg_name="Sticky-C1")

## Computing predicted labels
y_pred = sls.predict()
y_pred[0:40]

## Computing the hypnogram
hypno = yasa.Hypnogram(y_pred)
hypno_int = yasa.hypno_str_to_int(y_pred)
hypno_up = yasa.hypno_upsample_to_data(hypno=hypno_int, sf_hypno=(1/30), data=data, sf_data=sf)

# Inspecting bandpower
data_filt = raw.get_data() * 1e6
data_c1_uV = data_filt[6]
bandpower_stages = yasa.bandpower(data_c1_uV, sf = 100, win_sec=4, relative=True, hypno=hypno_up, include=(0, 1, 2, 3, 4))
bandpower_avg = bandpower_stages.groupby('Stage')[['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma']].mean()
bandpower_avg.index = ['Wake', 'N1', 'N2', 'N3', 'REM']

print(bandpower_avg)

# Spectrogram and hypnogram
print(hypno_up.shape, 'Unique values =', np.unique(hypno_up))
fig = yasa.plot_spectrogram(data_c1_uV, sf, hypno=hypno_up, fmax=30, cmap='Spectral_r', trimperc=5)
fig.show()

# Slow wave and sleep spindle detection
## Slow waves
sw = yasa.sw_detect(data_c1_uV, sf, hypno=hypno_up)
sw.summary()

sw.plot_average(time_before=0.4, time_after=0.8, center="NegPeak");
plt.legend(['C1'])
plt.show()

## Sleep spindles
sp = yasa.spindles_detect(data_c1_uV, sf)
sp.summary()

sp.plot_average(time_before=0.6, time_after=0.6);
plt.legend(['C1'])
plt.show()