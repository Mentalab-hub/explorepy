P300 Evoked Potential example project
====================================
This folder contains codes and instructions of a P300 Evoked Potential example project.

Experiment setup
----------------
* Follow the instructions for installing Explorepy in the
[documentation](https://explorepy.readthedocs.io/en/latest/installation.html#how-to-install). Use option 2 (via PyPI).
* Activate your Anaconda virtual environment.
* Install required packages by running this command:
`pip install matplotlib psychopy mne`
* Download the code from this page.
* In Conda terminal, navigate to `p300_demo` folder in the example directory of the Explorepy's code (e.g.
`cd <YOUR-FOLDER>\explorepy-master\examples\p300_demo`).
* Setup the cap and electrodes. Place EEG electrodes on the desired positions and the
ground electrode on Mastoid (or any other location far enough from other electrodes). TP10 and TP9 is used as ground location for  8 channel and 4 channel analysis in this example. Please refer to channel specific Python scripts(analysis_csv_X_channel.py) for channel locations.

Experiment
----------
In this experiment, we implement a simple visual oddball experiment with two visual stimuli,
a blue rectangle and a red oval as the standard and the target stimuli respectively. The following figure illustrates
an example trial presented to the subject. The subject is asked to press the space button whenever the red stimulus is displayed.

![alt text](exp.jpg "Visual oddball paradigm - an example trial")

Run the experiment by (put the device name and the desired file name in the command):
`python experiment.py -n Explore_#### -f rec_file_name`

Make sure the device is on and in advertising mode before running the command. The experiment has 10 blocks and
there are 5 trials in each block (50 trials in total). The numbers can be changed in `experiment.py` script (`n_blocks` and
`n_trials_per_block` variables).

| Variable           | Default value |
|--------------------|---------------|
| Stimulus duration  | 500 ms        |
| Number of blocks   | 10            |
| Trials per block   | 5             |
| Sampling rate      | 250           |
| Number of channels | 8 or 4        |
| Target marker code | 11            |
|Non-target marker code|10           |


When the experiment is completed, you will find three csv files (ExG, ORN and markers) in the working directory.

Running following command in the terminal will analyse the recorded data and generates the plots:\
8 channel: `python analysis_csv_8_channel.py -f rec_file_name`\
4 channel: `python analysis_csv_4_channel.py -f rec_file_name`

Results
-------
After running the analysis script, you should have a similar plot containing P300 waveforms for 8 channel devices.

![alt text](plots.jpeg "P300 plots")


Feel free to contact support[at]mentalab.com if you have any questions.
