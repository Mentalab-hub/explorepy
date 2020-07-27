%module  serialport

%{
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "ExploreException.h"
#include "BluetoothWorker.h"

extern "C"{
    #include <stdio.h>
    #include <errno.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <stdlib.h>
    #include <signal.h>
    #include <termios.h>
    #include <sys/poll.h>
    #include <sys/ioctl.h>
    #include <sys/socket.h>
    #include <sys/types.h>
    #include <assert.h>
}

#import <Foundation/NSObject.h>
#import <IOBluetooth/objc/IOBluetoothDevice.h>
#import <IOBluetooth/objc/IOBluetoothDeviceInquiry.h>
#import "pipe.h"

#include "BTSerialPortBinding.h"
%}

%include "cstring.i"
%cstring_output_withsize(char *bt_buffer, int* bt_length)

%typemap(in) (const char *write_buffer, int length) {
    Py_ssize_t len;
    PyBytes_AsStringAndSize($input, &$1, &len);
    $2 = (int)len;
}
%include "BTSerialPortBinding.h"
