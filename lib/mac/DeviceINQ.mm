#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "ExploreException.h"
#include "ExploreSDK.h"
#include <sys/stat.h>
#include <errno.h>

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
    #include <assert.h>
    #include <time.h>
    #include <dlfcn.h>
}

#import <Foundation/NSObject.h>
#import <IOBluetooth/objc/IOBluetoothDevice.h>
#import <IOBluetooth/objc/IOBluetoothRFCOMMChannel.h>
#import <IOBluetooth/objc/IOBluetoothSDPUUID.h>
#import <IOBluetooth/objc/IOBluetoothSDPServiceRecord.h>
#import "BluetoothWorker.h"
#include "json.hpp"

namespace {
    std::string getSharedObjectFileForObject(const void *obj)
    {
        Dl_info dlInfo{};
        if (dladdr(obj, &dlInfo) == 0)
            return "";
        return dlInfo.dli_fname;
    }

    NSURL* urlForThisFile()
    {
        NSURL* myPath = [[[NSURL alloc]
            initFileURLWithPath: [[NSString alloc]
                initWithUTF8String: getSharedObjectFileForObject((void*)&urlForThisFile).c_str()]]
                    URLByStandardizingPath];
        return myPath;
    }
}

ExploreSDK *ExploreSDK::Create()
{
	return new ExploreSDK();
}

ExploreSDK::ExploreSDK()
{
}

ExploreSDK::~ExploreSDK()
{
}

std::vector<device> ExploreSDK::PerformDeviceSearch(int)
{
    // The helper executable should be in the same directory as this shared object.
    NSURL* myPath = urlForThisFile();
    NSURL* helperExecutable = [[myPath URLByDeletingLastPathComponent] URLByAppendingPathComponent: @"btScan"];
    
    //changing the file permission using C function chmod()
    NSString *nsString = helperExecutable.path;
    const char* convertedUrl = [nsString UTF8String];
    
    if(chmod(convertedUrl , S_IXUSR) != 0){
        printf("Error occurred while changing file permission :'%s '\n", strerror(errno));
    }
	
    NSPipe* pipe = [[NSPipe alloc] init];

    NSTask* task = [[NSTask alloc] init];
    [task setStandardOutput: pipe];
    [task setExecutableURL: helperExecutable];

    NSError* e = nil;
    if (![task launchAndReturnError: &e]) {
        printf("%s\n", [[e localizedDescription] UTF8String]);
    }
    NSFileHandle* fh = pipe.fileHandleForReading;
    NSData* data = [fh readDataToEndOfFile];
    std::string_view outText((char *)data.bytes, data.length);
    nlohmann::json jResult = nlohmann::json::parse(outText);

    std::vector<device> devices;
    for (auto& j : jResult) {
        device dev;
		dev.address = j["addressString"];
		dev.name = j["nameOrAddress"];
		dev.connected = j["isConnected"];
		dev.remembered = j["isFavorite"];
		dev.authenticated = j["isPaired"];
		dev.lastSeen = (uint32_t)j["lastSeen"];
		dev.lastUsed = 0;

        int cod = j["classOfDevice"];
		//dev.deviceClass = (DeviceClass)(cod & 0x1ffc);
		//dev.majorDeviceClass = (DeviceClass)(cod & DC_Uncategorized);
	    //dev.serviceClass = (ServiceClass)(cod >> 13);
		devices.push_back(dev);
    }
    return devices;
}

int ExploreSDK::SdpSearch(std::string address)
{
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    NSString *addr = [NSString stringWithCString: address.c_str() encoding: NSASCIIStringEncoding];
    BluetoothWorker *worker = [BluetoothWorker getInstance];
    int channelID = [worker getRFCOMMChannelID: addr];

    [pool release];
	return channelID;
}
