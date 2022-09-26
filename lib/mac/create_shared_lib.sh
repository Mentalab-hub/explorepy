#swig -python -c++ -py3 -extranative -debug-classes swig_interface.i
#c++ -c -fpic swig_interface_wrap.cxx -I/usr/include/python3.6

#c++ -c -fpic BTSerialPortBinding.cpp DeviceINQ.cpp -I/usr/include/python3.6

#c++ -shared BTSerialPortBinding.o DeviceINQ.o swig_interface_wrap.o -lbluetooth -o _exploresdk.so


#CPPFLAGS="-std=c++11" pip install -e .

#mac installation instructions and troubleshooting:
#xcode installation
#xcode-select --install
#accept licence:
#sudo xcodebuild -license
#xcode-select --install # Install Command Line Tools if you haven't already.
#sudo xcode-select --switch /Library/Developer/CommandLineTools # Enable command line tools
#then just ignore the warning.

#Alternatively if you want you can use the full Xcode.app (if you have it installed) with:

# Change the path if you installed Xcode somewhere else.
#sudo xcode-select -s /Applications/Xcode.app/Contents/Developer

# switch python versions:
#ls -l /usr/local/bin/python*
#ln -s -f /usr/local/bin/python3.7 /usr/local/bin/python
#python can be found at:
#/Library/Frameworks/Python.framework/Versions/3.7/

swig -python -c++ -py3 -extranative -threads -debug-classes   swig_interface.i
# for windows: use the -threads option 
#swig -python -c++ -py3 -extranative -debug-classes swig_interface.i
c++ -c -fpic swig_interface_wrap.cxx -I/usr/local/Cellar/python@3.7/3.7.11/Frameworks/Python.framework/Versions/3.7/include/python3.7m/ -ObjC++ -std=c++11

c++ -c -fpic BluetoothDeviceResources.mm -I/usr/local/Cellar/python@3.7/3.7.11/Frameworks/Python.framework/Versions/3.7/include/python3.7m/ -ObjC++ -std=c++11

c++ -c -fpic BluetoothWorker.mm -I/usr/local/Cellar/python@3.7/3.7.11/Frameworks/Python.framework/Versions/3.7/include/python3.7m/ -ObjC++ -std=c++11

c++ -c -fpic BTSerialPortBinding.mm -I/usr/local/Cellar/python@3.7/3.7.11/Frameworks/Python.framework/Versions/3.7/include/python3.7m/ -ObjC++ -std=c++11

c++ -c -fpic -std=c++11 DeviceINQ.mm -I/usr/local/Cellar/python@3.7/3.7.11/Frameworks/Python.framework/Versions/3.7/include/python3.7m/ -ObjC++ -std=c++11

gcc  -c -fpic pipe.c -I/usr/local/Cellar/python@3.7/3.7.11/Frameworks/Python.framework/Versions/3.7/include/python3.7m/

c++ -shared -flat_namespace -undefined suppress BTSerialPortBinding.o DeviceINQ.o BluetoothDeviceResources.o BluetoothWorker.o pipe.o  swig_interface_wrap.o -std=c++11  -framework foundation -framework IOBluetooth -o _exploresdk.so

rm -rf *.o
