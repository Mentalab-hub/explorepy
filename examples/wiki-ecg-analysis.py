import neurokit2 as nk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyedflib

def edf_to_arr(edf_path):
    f = pyedflib.EdfReader(edf_path)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
    
    return sigbufs

data = edf_to_arr("../data/wiki-ECG-resting_ExG.bdf") # read in bdf file
data = data[1] # channel 1 holds ECG data
data *= 1e-3 # scale uV to mV for ECG analysis in neurokit
ecg_signals, info = nk.ecg_process(data[5000:25000], sampling_rate=250) # convert 

rpeaks = info["ECG_R_Peaks"]
cleaned_ecg = ecg_signals["ECG_Clean"]

intervalrelated = nk.ecg_intervalrelated(ecg_signals)
intervalrelated.iloc[0,1:83]

nk.ecg_plot(ecg_signals, info)
fig = plt.gcf()
fig.set_size_inches(20, 12, forward=True)
fig.savefig("../plots/resting_ecg.png")

ecg_signals, info = nk.ecg_process(data[18000:20000] , sampling_rate=250) # take a subset from the middle of recording and apply neurokit
rpeaks = info["ECG_R_Peaks"]
cleaned_ecg = ecg_signals["ECG_Clean"]
plot = nk.events_plot(rpeaks, cleaned_ecg[0:cleaned_ecg.shape[0]])
fig = plt.gcf()
fig.set_size_inches(20, 12, forward=True)
fig.savefig("../plots/resting_rpeaks.png")

peaks, info = nk.ecg_peaks(data[5000:25000], sampling_rate=250)
hrv_time = nk.hrv_time(peaks, sampling_rate=250, show=True)
hrv_time
fig = plt.gcf()
fig.set_size_inches(20, 12, forward=True)
fig.savefig("../plots/sv_ecg_time.png")