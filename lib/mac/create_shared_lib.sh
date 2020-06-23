swig -python -c++ -py3 -extranative -debug-classes swig_interface.i
c++ -c -fpic swig_interface_wrap.cxx -I/usr/include/python3.6

c++ -c -fpic BTSerialPortBinding.cpp DeviceINQ.cpp -I/usr/include/python3.6

c++ -shared BTSerialPortBinding.o DeviceINQ.o swig_interface_wrap.o -lbluetooth -o _exploresdk.so
