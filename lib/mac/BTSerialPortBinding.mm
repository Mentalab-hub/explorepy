//iostream for debug

#include<iostream>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "ExploreException.h"
#include "BTSerialPortBinding.h"
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

struct bluetooth_data
{
	pipe_consumer_t *consumer;
};

size_t size_buffer;
bool isSocketClosed = false;

BTSerialPortBinding *BTSerialPortBinding::Create(string address, int channelID)
{
		//throw ExploreException("ChannelID should be a positive int value");

	return new BTSerialPortBinding(address, channelID);
}

BTSerialPortBinding::BTSerialPortBinding(string address, int channelID)
	: address(address), channelID(channelID), data(new bluetooth_data())
{
	data->consumer = NULL;
}

BTSerialPortBinding::~BTSerialPortBinding()
{
}

int BTSerialPortBinding::Connect()
{
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    NSString *addressString = [NSString stringWithCString:address.c_str() encoding:NSASCIIStringEncoding];
    BluetoothWorker *worker = [BluetoothWorker getInstance];
    // create pipe to communicate with delegate

    pipe_t *pipe = pipe_new(sizeof(unsigned char), 0);
	int status;

    IOReturn result = [worker connectDevice: addressString onChannel:channelID withPipe:pipe];

    if (result == kIOReturnSuccess) {
        pipe_consumer_t *c = pipe_consumer_new(pipe);

        // save consumer side of the pipe
        data->consumer = c;
        status = 0;
    } else {
        status = 1;
    }

    pipe_free(pipe);
    [pool release];


    return status;
}

void BTSerialPortBinding::Close()
{
    NSString *addressString = [NSString stringWithCString:address.c_str() encoding:NSASCIIStringEncoding];
    BluetoothWorker *worker = [BluetoothWorker getInstance];
    [worker disconnectFromDevice: addressString];
}

void BTSerialPortBinding::Read(char *buffer, int *length)
{
    if (data->consumer == NULL){
        isSocketClosed = true;
        cout << "Socket closed as data consumer is null" << endl;
        throw ExploreIOException("BT socket is closed!");
    }
    

	if (buffer == nullptr)
	return;

    size_buffer = -1;
    
    size_buffer = pipe_pop_eager(data->consumer, buffer, *length);
    
    if (size_buffer == 0) {
        pipe_consumer_free(data->consumer);
        data->consumer = NULL;
        cout << "Possible socket closure, raising IO Exception" << endl;
        throw ExploreIOException("BT socket is closed!");
        
        
    }
    if(size_buffer < *length){
        if(isSocketClosed){
            cout << "Socket closed from read command read function..length" << endl;
            throw ExploreIOException("BT socket is closed!");
        }
        else{
            cout << "Read error from read command read function..normal operation...length" << endl;
            throw ExploreReadBufferException("EMPTY_BUFFER_ERROR");
        }
    }

    // when no data is read from rfcomm the connection has been closed.
    
}

void BTSerialPortBinding::Write(const char *buffer, int length)
{
	if (buffer == nullptr)
	//throw ExploreException("buffer cannot be null");
	return;


	if (length == 0)
		return;

    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    BluetoothWorker *worker = [BluetoothWorker getInstance];
    NSString *addressString = [NSString stringWithCString:address.c_str() encoding:NSASCIIStringEncoding];

    if ([worker writeAsync: const_cast<char*>(buffer) length: length toDevice: addressString] != kIOReturnSuccess)
    return;

    [pool release];
}

bool BTSerialPortBinding::IsDataAvailable()
{
	return false;
}
