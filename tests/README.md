# Unit tests
This subfolder contains unit tests for the explorepy API.
Test resources like input and output files are found in `tests/res/in/` and `tests/res/out/` respectively.
Input files are saved as binary files, output files are saved as text files in json format.

## test_packet.py
Some of the tests in this file fail since the source file describes abstract classes that aren't actually abstract.
