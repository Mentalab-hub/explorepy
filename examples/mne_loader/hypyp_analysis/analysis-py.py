import copy
import matplotlib
matplotlib.use("QtAgg")                        # choose Qt backend
import matplotlib.pyplot as plt
import mne
from mne import export
from hypyp import prep, analyses, stats, viz
from collections import OrderedDict

from statsmodels.compat import scipy
import numpy as np

from mne_icalabel import label_components

def ICA_autocorrect(icas: list, epochs: list, verbose: bool = False) -> list:
    """
    Automatically detect the ICA components that are not brain related and remove them.

    Arguments:
        icas: list of Independent Components for each participant (IC are MNE
          objects).
        epochs: list of 2 Epochs objects (for each participant). Epochs_S1
          and Epochs_S2 correspond to a condition and can result from the
          concatenation of Epochs from different experimental realisations
          of the condition.
          Epochs are MNE objects: data are stored in an array of shape
          (n_epochs, n_channels, n_times) and parameters information is
          stored in a disctionnary.
        verbose: option to plot data before and after ICA correction,
          boolean, set to False by default.

    Returns:
        cleaned_epochs_ICA: list of 2 cleaned Epochs for each participant
          (the non-brain related IC have been removed from the signal).
    """

    cleaned_epochs_ICA = []
    for ica, epoch in zip(icas, epochs):
        ica_with_labels_fitted = label_components(epoch, ica, method="iclabel")
        ica_with_labels_component_detected = ica_with_labels_fitted["labels"]
        # Remove non-brain components (take only brain components for each subject)
        excluded_idx_components = [idx for idx, label in enumerate(ica_with_labels_component_detected) if label not in ["brain"]]
        cleaned_epoch_ICA = mne.Epochs.copy(epoch)
        cleaned_epoch_ICA.info['bads'] = []
        ica.apply(cleaned_epoch_ICA, exclude=excluded_idx_components)
        cleaned_epoch_ICA.info['bads'] = copy.deepcopy(epoch.info['bads'])
        cleaned_epochs_ICA.append(cleaned_epoch_ICA)

        if verbose:
            epoch.plot(title='Before ICA correction')
            cleaned_epoch_ICA.plot(title='After ICA correction')
    return cleaned_epochs_ICA

# Reading joing tapping recording
raw1 = mne.io.read_raw_fif(fname='salman_sync_raw.fif', preload=True)
raw1.notch_filter(50)
raw1.filter(1, 100)
epo1 = mne.make_fixed_length_epochs(
    raw1, duration=2.0, overlap=1.5, preload=True, reject_by_annotation=True
)

raw2 = mne.io.read_raw_fif(fname='christoph_sync_raw.fif', preload=True)
raw2.notch_filter(50)
raw2.filter(1, 100)
epo2 = mne.make_fixed_length_epochs(
    raw2, duration=2.0, overlap=1.5, preload=True, reject_by_annotation=True
)


# Define frequency bands as a dictionary
freq_bands = {
    'Alpha-Low': [7.5, 11],
    'Alpha-High': [11.5, 13]
}

# Convert to an OrderedDict to keep the defined order
freq_bands = OrderedDict(freq_bands)
print('Frequency bands:', freq_bands)

mne.epochs.equalize_epoch_counts([epo1, epo2])
sampling_rate = epo1.info['sfreq']


print('')


icas = prep.ICA_fit([
    epo1, epo2
],
    n_components=15,
    method='infomax',
    fit_params=dict(extended=True),
    random_state=42
)

#
cleaned_epochs_ICA = ICA_autocorrect(icas, [epo1, epo2], verbose=False)


cleaned_epochs_AR, dic_AR = prep.AR_local(
    cleaned_epochs_ICA,
    strategy="union",
    threshold=50.0,
    verbose=True
)
print('AutoReject completed.')
# # Assign cleaned epochs to individual participant variables
preproc_S1 = cleaned_epochs_AR[0]
preproc_S2 = cleaned_epochs_AR[1]
# print('Preprocessed epochs for both participants are ready.')
#
#
# # Compute PSD for participant 1 in the Alpha-Low band
# psd1 = analyses.pow(
#     preproc_S1,
#     fmin=7.5,
#     fmax=11,
#     n_fft=1000,
#     n_per_seg=1000,
#     epochs_average=True
# )
#
# # Compute PSD for participant 2 in the Alpha-Low band
# psd2 = analyses.pow(
#     preproc_S2,
#     fmin=7.5,
#     fmax=11,
#     n_fft=1000,
#     n_per_seg=1000,
#     epochs_average=True
# )
#
# # Combine PSD data into a single array
# data_psd = np.array([psd1.psd, psd2.psd])
# print('PSD analysis completed.')
#
# # Prepare data for connectivity analysis (combine both participants)
data_inter = np.array([preproc_S1, preproc_S2])
result_intra = []
#
# # Compute the analytic signal in each frequency band
complex_signal = analyses.compute_freq_bands(
    data_inter,
    sampling_rate,
    freq_bands,
    filter_length=int(sampling_rate),  # Adjust filter length based on sampling rate
    l_trans_bandwidth=5.0,  # Reduced transition bandwidth
    h_trans_bandwidth=5.0
)

# Compute connectivity using cross-correlation ('ccorr') and average across epochs
result = analyses.compute_sync(complex_signal, mode='ccorr', epochs_average=True)

# Determine the number of channels
n_ch = len(epo1.info['ch_names'])

# Slice the connectivity matrix to get inter-brain connectivity in the Alpha-Low band
alpha_low, alpha_high = result[:, 0:n_ch, n_ch:2 * n_ch]

# For further analysis, choose the Alpha-Low band values
values = alpha_low

# Compute a Z-score normalized connectivity matrix
C = (values - np.mean(values[:])) / np.std(values[:])

# Process intra-brain connectivity for each participant
for i in [0, 1]:
    # Slice intra-brain connectivity matrix
    alpha_low, alpha_high = result[:, (i * n_ch):((i + 1) * n_ch), (i * n_ch): ((i + 1) * n_ch)]
    values_intra = alpha_low

    # Remove self-connections
    values_intra -= np.diag(np.diag(values_intra))

    # Compute Z-score normalization for intra connectivity
    C_intra = (values_intra - np.mean(values_intra[:])) / np.std(values_intra[:])
    result_intra.append(C_intra)

print('Connectivity analysis completed.')
#
#
# # Compute mean PSD values for each channel across epochs for both participants
# psd1_mean = np.mean(psd1.psd, axis=1)
# psd2_mean = np.mean(psd2.psd, axis=1)
#
# # Combine the means into a single array for the t-test
# X = np.array([psd1_mean, psd2_mean])
#
# # Perform permutation t-test (using MNE) without correction for multiple comparisons
# T_obs, p_values, H0 = mne.stats.permutation_t_test(
#     X=X,
#     n_permutations=5000,
#     tail=0,
#     n_jobs=1
# )
# print('Permutation t-test completed.')
#
# # Alternatively, compute statistical conditions using HyPyP's statsCond function
# statsCondTuple = stats.statsCond(
#     data=data_psd,
#     epochs=preproc_S1,
#     n_permutations=5000,
#     alpha=0.05
# )
# print('Statistical condition tuple computed.')
#
#
# # Create connectivity matrix for a priori sensor connectivity using participant 1's sensor layout
# con_matrixTuple = stats.con_matrix(preproc_S1, freqs_mean=psd1.freq_list)
# ch_con_freq = con_matrixTuple.ch_con_freq
#
# # Create two fake groups by replicating the PSD data and adding a small noise
# noise_level = 1e-6  # Small noise to break exact duplicates
# data_group = [
#     np.array([psd1.psd + np.random.normal(0, noise_level, psd1.psd.shape) for _ in range(3)]),
#     np.array([psd2.psd + np.random.normal(0, noise_level, psd2.psd.shape) for _ in range(3)])
# ]
#
# # Perform non-parametric cluster-based permutation test on the fake groups
# statscondCluster = stats.statscondCluster(
#     data=data_group,
#     freqs_mean=psd1.freq_list,
#     ch_con_freq=scipy.sparse.bsr_matrix(ch_con_freq),
#     tail=1,
#     n_permutations=5000,
#     alpha=0.05
# )
# print('Cluster-based permutation test for PSD completed.')
#
#
# # Create connectivity matrix for intra-brain connectivity
# con_matrixTuple = stats.con_matrix(
#     epochs=preproc_S1,
#     freqs_mean=np.arange(7.5, 11),
#     draw=False
# )
#
# ch_con = con_matrixTuple.ch_con
#
# # Create fake groups for intra-brain connectivity analysis
# Alpha_Low = [
#     np.array([
#         result_intra[0] + np.random.normal(0, noise_level, result_intra[0].shape),
#         result_intra[0] + np.random.normal(0, noise_level, result_intra[0].shape)
#     ]),
#     np.array([
#         result_intra[1] + np.random.normal(0, noise_level, result_intra[1].shape),
#         result_intra[1] + np.random.normal(0, noise_level, result_intra[1].shape)
#     ])
# ]
#
# # Run cluster-based permutation test for intra-brain connectivity
# statscondCluster_intra = stats.statscondCluster(
#     data=Alpha_Low,
#     freqs_mean=np.arange(7.5, 11),
#     ch_con_freq=scipy.sparse.bsr_matrix(ch_con),
#     tail=1,
#     n_permutations=5000,
#     alpha=0.05
# )
# print('Intra-brain connectivity cluster test completed.')
#
#
# # ===================================================================
# # now, test with a fake group
# # Create fake groups for inter-brain connectivity analysis
# data = [
#     np.array([
#         values,
#         values + np.random.normal(0, 1e-6, values.shape)
#     ]),
#     np.array([
#         result_intra[0],
#         result_intra[0] + np.random.normal(0, 1e-6, result_intra[0].shape)
#     ])
# ]
#
# print(len(data[0][0]), len(data[0][1]), len(data[1][0]), len(data[1][1]))
#
#
# # Run cluster-based permutation test for inter-brain connectivity without connectivity priors
# statscondCluster = stats.statscondCluster(
#     data=data,
#     freqs_mean=np.linspace(7.5, 11, data[0].shape[-1]),
#     ch_con_freq=None,
#     tail=0,
#     n_permutations=5000,
#     alpha=0.05
# )
# print('[Fake group] Inter-brain connectivity cluster test completed.')
#
# # ==============================================================================
# # Plotting =====================================================================
#
#
# # Plot sensor-level T-values using the t-statistics computed earlier
# viz.plot_significant_sensors(
#     T_obs_plot=statsCondTuple.T_obs,
#     epochs=preproc_S1
# )
# print('Sensor-level T-values plotted.')
#
# # Plot only the T-values for sensors that are statistically significant
# viz.plot_significant_sensors(
#     T_obs_plot=statsCondTuple.T_obs_plot,
#     epochs=preproc_S1
# )
# print('Significant sensors T-values plotted.')


plt.ion()
# ax = viz.viz_3D_intra(epo1, epo2,
#                  C1= result_intra[0],
#                  C2= result_intra[1],
#                  threshold= 0.8,
#                  steps=10,
#                  lab=False,
#                 )

viz.viz_2D_topomap_inter(epo1, epo2, C, threshold=2, steps=10, lab=True)
plt.show(block=True)

