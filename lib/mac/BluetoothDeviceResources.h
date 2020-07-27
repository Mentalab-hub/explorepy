#ifndef NODE_BTSP_SRC_BLUETOOTH_DEVICE_RESOURCES_H
#define NODE_BTSP_SRC_BLUETOOTH_DEVICE_RESOURCES_H

#import <Foundation/NSObject.h>
#import <IOBluetooth/objc/IOBluetoothDevice.h>
#import <IOBluetooth/objc/IOBluetoothRFCOMMChannel.h>
#import "pipe.h"

@interface BluetoothDeviceResources: NSObject {
    pipe_producer_t *producer;
    IOBluetoothDevice *device;
    IOBluetoothRFCOMMChannel *channel;
}

@property (readwrite, assign) pipe_producer_t *producer;
@property (nonatomic, retain) IOBluetoothDevice *device;
@property (nonatomic, retain) IOBluetoothRFCOMMChannel *channel;

@end

#endif
