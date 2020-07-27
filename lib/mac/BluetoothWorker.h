#ifndef NODE_BTSP_SRC_BLUETOOTH_WORKER_H
#define NODE_BTSP_SRC_BLUETOOTH_WORKER_H

#include <string>
#import <Foundation/NSObject.h>
#import <IOBluetooth/objc/IOBluetoothRFCOMMChannel.h>
#import <IOBluetooth/objc/IOBluetoothDevice.h>
#import <IOBluetooth/objc/IOBluetoothDeviceInquiry.h>

#import "pipe.h"

struct device_info_t {
	std::string address;
	std::string name;
	bool connected;
	bool paired;
	bool favorite;
	int classOfDevice;
	int rssi;
	double lastSeen;
};

@interface BluetoothWorker: NSObject<IOBluetoothRFCOMMChannelDelegate> {
    @private
	NSMutableDictionary *devices;
    NSThread *worker;
    pipe_producer_t *inquiryProducer;
	NSLock *sdpLock;
	NSLock *connectLock;
	NSLock *devicesLock;
	IOReturn connectResult;
	int lastChannelID;
	NSLock *writeLock;
	IOReturn writeResult;
	NSTimer *keepAliveTimer;
}

+ (id)getInstance;
- (void) disconnectFromDevice: (NSString *) address;
- (IOReturn)connectDevice: (NSString *) address onChannel: (int) channel withPipe: (pipe_t *)pipe;
- (IOReturn)writeAsync:(void *)data length:(UInt16)length toDevice: (NSString *)address;

- (int) getRFCOMMChannelID: (NSString *) address;

- (void)rfcommChannelData:(IOBluetoothRFCOMMChannel*)rfcommChannel data:(void *)dataPointer length:(size_t)dataLength;
- (void)rfcommChannelClosed:(IOBluetoothRFCOMMChannel*)rfcommChannel;

- (void) rfcommChannelWriteComplete:(IOBluetoothRFCOMMChannel*)rfcommChannel refcon:(void*)refcon status:(IOReturn)error;

@end

#endif
