%module  serialport

%{
#include <winsock2.h>
#include <windows.h>
#include <ws2bth.h>
#include <string>
#include <stdlib.h>
#include "ExploreException.h"
#include "BTSerialPortBinding.h"
#include "BluetoothHelpers.h"
#include <string>
#include <memory>
#include "ExploreExceptionConstants.h"

#include "BTSerialPortBinding.h"
%}

%include "cstring.i"
%include "typemaps.i"


%cstring_output_withsize(char *bt_buffer, int* bt_length)

%typemap(in) (const char *write_buffer, int length) {
    Py_ssize_t len;
    PyBytes_AsStringAndSize($input, &$1, &len);
    $2 = (int)len;
}

%include "BTSerialPortBinding.h"


