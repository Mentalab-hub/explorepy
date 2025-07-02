% --- CSV FILE LOADING ---

% Prompt user for CSV file name
filename = input('Enter CSV filename: ', 's');

% Prompt user for metadata file name
meta_filename = input('Enter metadata filename: ', 's');

% Load CSV
csv_data = readmatrix(filename);
time = csv_data(:,1);
eeg  = csv_data(:,2:end);

% Load meta data and get sampling rate
meta_data = readtable(meta_filename);
sampling_rate = meta_data.sr(1);

% Build FieldTrip data structure
data_csv = [];
data_csv.fsample = sampling_rate;
data_csv.time = {time' - time(1)};
data_csv.trial = {eeg'};
data_csv.label = arrayfun(@(x) sprintf('ch%d', x), 1:size(eeg,2), 'UniformOutput', false);

fprintf('Sampling rate (fsample): %.2f Hz\n', data_csv.fsample);

% Visualize
cfg = [];
cfg.viewmode = 'vertical';
ft_databrowser(cfg, data_csv);