Steady State Visual Evoked Potential (SSVEP) demo
====================================


In this example project, we have implemented a simple SSVEP experiment using Mentalab's Explore device.
Four flickering stimuli are shown on the screen with the following frequencies (assuming a screen with 60 Hz refresh rate).
* Bottom left: 12 Hz
* Up left: 10 Hz
* Up right: 8.6 Hz
* Bottom right: 7.5 Hz

The predicted target based on subject's EEG is shown in the center.

![alt text](ssvep.jpg "Screenshot of SSVEP experiment")

Requirements
------------
* [explorepy](https://github.com/Mentalab-hub/explorepy)
* [scikit-learn](https://github.com/scikit-learn/scikit-learn)
* [pyschopy](https://github.com/psychopy/psychopy)
* [matplotlib](https://github.com/matplotlib/matplotlib)

Usage
-----
Place EEG electrodes on the occipital visual cortex (Oz, O1, O2, POz, etc.) and the ground electrode on Fpz
(or any other location far enough from other electrodes). Turn on the device and run the following command
in your terminal.

```
$ python main.py --name Explore_1438 --duration 100
```
