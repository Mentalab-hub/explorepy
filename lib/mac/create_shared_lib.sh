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

#For Sonoma OS: 
#/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Headers
swig -python -c++ -py3 -extranative -threads -debug-classes   swig_interface.i
# for windows: use the -threads option
#swig -python -c++ -py3 -extranative -debug-classes swig_interface.i
c++ -c -fpic swig_interface_wrap.cxx -I/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Headers -ObjC++ -std=c++11

c++ -c -fpic BluetoothDeviceResources.mm -I/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Headers -ObjC++ -std=c++11

c++ -c -fpic BluetoothWorker.mm -I/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Headers -ObjC++ -std=c++11

c++ -c -fpic BTSerialPortBinding.mm -I/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Headers -ObjC++ -std=c++11

c++ -c -fpic -std=c++11 DeviceINQ.mm -I/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Headers -ObjC++ -std=c++11

gcc  -c -fpic pipe.c -I/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Headers

# Get the architecture
architecture=$(arch)

# Base command
base_command="c++ -shared -flat_namespace"

# Add architecture-specific flag if architecture is arm64
if [ "$architecture" == "arm64" ]; then
    arch_flag="-arch arm64"
else
    arch_flag=""
fi

# Complete command with other flags and files
complete_command="$base_command $arch_flag -undefined suppress BTSerialPortBinding.o DeviceINQ.o BluetoothDeviceResources.o BluetoothWorker.o pipe.o swig_interface_wrap.o -std=c++11 -framework foundation -framework IOBluetooth -o _exploresdk.so"

# Execute the complete command
$complete_command

rm -rf *.o
