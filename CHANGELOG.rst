
Changelog
=========

4.1.0 (31.3.2025)
------------------
* Improve bluetooth connection UX
* Add more data in IMU packet
* Improve USB streaming


4.0.0 (14.3.2025)
------------------
* Deprecate support for Explore legacy devices
* Improve binary file conversion and stream parser algorithm


3.2.1 (31.1.2025)
------------------
* Hotfix impedance calculation


3.2.0 (31.1.2025)
------------------
* Fix critical bug in settings file modification
* Improve BT communication
* Disable impedance on fresh connection
* Introduce new sampling rates


3.1.0 (8.11.2024)
------------------
* Add USB streaming support
* Improve BT communication


3.0.1 (28.08.2024)
------------------
* Improve device connectivity
* Add text marker support


3.0.0 (7.6.2024)
------------------
* Improve Mac OSX support
* Add Support for Explore Pro
* Robust data parsing


2.0.0 (5.1.2024)
------------------
* Add Mac OSX support
* Remove browser based visualization support
* Update build script for docs


1.8.2 (28.8.2023)
------------------
* Improve explore mock device
* Add yaml file for RTD build


1.8.1 (11.7.2023)
------------------
* Bugfix record data
* Add dummy packets for CI/CD pipeline
* Add mock device


1.8.0 (27.2.2023)
------------------
* Add support for explore+ 16 ch device
* Unit test integration with CI/CD
* Repair CSV from binary data
* Save current device settings on disconnection, restore them on demand
* Receive external triggers from LSL


1.7.0 (21.12.2022)
------------------
* Add support for new explore+ 32 ch device
* Sorted timestamps in CSV
* Settings file to preserve experiment settings


1.6.3 (25.10.2022)
------------------
* Add new 8 channel Explore+ device


1.6.2 (7.09.2022)
------------------
* Change EDF file extension
* Add dataset export feature for EEGLab
* Bugfix recording


1.6.1 (14.06.2022)
------------------
* Improve EDF file timing with PyEDFlib
* Fix Bokeh import error
* Add custom analysis script for 4 channel P300 experiment


1.6.0 (23.11.2021)
------------------
* Remove dependency on Pybluez
* Add SSVEP offline experiment
* Add P300 example
* Update LSL stream names
* Fixing some bugs


1.5.2 (22.09.2021)
------------------
* Hotfix for impedance disable bug


1.5.1 (21.7.2021)
------------------
* Hotfix for installation bug

1.5.0 (20.7.2021)
------------------
* Push to LSL button in the GUI
* Search free port for initialization of the dashboard
* Update installation procedure for Windows
* Fixing some minor bugs

1.4.0 (31.03.2021)
------------------
* Baseline correction feature in the visualization
* Error logging system (Logging and Sentry)
* Fix a bug of SDK in MacOS
* Change the default Bluetooth interface to SDK
* Improved FFT visualization
* More precise local time in all OSes


1.3.0 (30-12-2020)
------------------
* GUI resizing for different screen resolutions
* Added event button to dashboard
* Improvement of impedance measurement
* Fixed minor bugs

1.2.0 (25-11-2020)
------------------
* Standalone installer for Windows OS
* Fix bugs in ExploreSDK
* Create new file on device settings change


1.1.0 (27-08-2020)
------------------
* MacOS support
* Fix bugs
* Add module disable/enable feature
* Enhanced user interface
* Add unique lsl stream names


1.0.0 (22-05-2020)
------------------
* Add ExploreSDK as the Bluetooth interface
* New dark theme
* Record module in dashboard
* Improvement in visualization perfrmance
* CLI migration to Click


0.6.0 (17-02-2020)
------------------
* EDF (BDF+) file writer
* Channel disable/enable feature
* Calibration of movement sensors
* Extraction of physical orientation (angle and rotation)
* Soft marker event
* Visualization performance enhancement
* Automatic number of channel and sampling rate detection
* Exception handling improvement
* Command for soft reset of Explore
* Marker visualization


0.5.0 (25-11-2019)
------------------
* Impedance measurement
* Send commands to device
* Configuring device settings
* Update push to lsl feature

0.4.0 (09-09-2019)
------------------
* Added marker feature
* Timer based recording
* Fixed a bug in csv file writer
* Fixed a bug in device reconnect
* Improved performance of dashboard visualization


0.3.1 (28-05-2019)
------------------
* Fixed a bug in 8-channel ExG packet conversion
* Fixed a minor bug in the record function
* Updated the documentation


0.3.0 (10-05-2019)
------------------
* Explore dashboard
* Real-time visualization of ExG and orientation signal
* Device information in Dashboard
* Environmental data (battery, temperature and light)
* Real-time bandpass filter
* New packet structures (ADS1294R & ADS1298R)
* Heart rate estimation and R-peaks detector in dashboard


0.2.0 (2019-03-08)
------------------

* Added real-time recording feature
* Added Command Line Interface
* Added lsl integration
* Added new packet classes
* Fixed reconnect issues
* Removed input requests inside functions


0.1.0 (2019-01-18)
------------------

* First release on PyPI.
