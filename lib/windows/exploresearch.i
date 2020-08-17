%module exploresearch

%{
#include "ExploreException.h"
#include "ExploreSDK.h"

#include <initguid.h>
#include <winsock2.h>
#include <windows.h>
#include <string>
#include <stdlib.h>

#include <ws2bth.h>
#include <memory>
#include <bluetoothapis.h>
#include "ExploreException.h"
#include "ExploreSDK.h"
#include "BluetoothHelpers.h"

%}

%template(vectordevice) std::vector<device>;


%include "ExploreException.h"
%include "ExploreSDK.h"
