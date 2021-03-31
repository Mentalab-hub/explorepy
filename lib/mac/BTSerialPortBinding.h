/**
 * @file BTSerialPortBinding.h
 * @brief Bluetooth serial port communication class to connect, read and write data
 *
 */

#pragma once

#include <string>
#include <memory>
#include "ExploreExceptionConstants.h"

using namespace std;
struct bluetooth_data;

class BTSerialPortBinding
{
private:
	std::string address;
	int channelID;
	std::unique_ptr<bluetooth_data> data;
	//int readTimeout;
	//int writeTimeout;
	BTSerialPortBinding(std::string address, int channelID);

public:

/**
@brief Destructor for BTSerialPortBinding, is called automatically

@param
@return
*/
	~BTSerialPortBinding();

/**
    Static function to initialize ExploreSDK instance

    @param address mac address of the bluetooth device
    @param channelID channel ID of the bluetooth device, can be retrieved after the search
    @return pointer to initialized BTSerialPortBinding class
*/

	static BTSerialPortBinding *Create(std::string address, int channelID);

/**
    Static function to initialize ExploreSDK instance

    @param
    @return int returns 0 on success, a negativve number otherwise
    @throw ExploreSDK Exception, please refer to ExploreSDK Constants fie for exceptions
*/
	int Connect();

/**
    Static function to initialize ExploreSDK instance

    @param
    @return void
    @throw ExploreSDK Exception, please refer to ExploreSDK Constants fie for exceptions
*/
	void Close();

/**
    Reads the data from the buffer

    @param buffer the buffer where the received data will be stored
    @param length the length of the buffer
    @return size of the buffer that has been filled with the received data
    @throw ExploreSDK Exception, please refer to ExploreSDK Constants fie for exceptions
*/
	void Read(char *bt_buffer, int* bt_length);

/**
    Sends data to the device

    @param buffer the data to send to the device
    @param length the length of buffer
    @return
    @throw ExploreSDK Exception, please refer to ExploreSDK Constants fie for exceptions
*/
	void Write(const char *write_buffer, int length);

/**
    Checks if data is available on the device buffer

    @param buffer the data to send to the device
    @param
    @return boolean
    @throw ExploreSDK Exception, please refer to ExploreSDK Constants fie for exceptions
*/
	bool IsDataAvailable();
	//void SetTimeouts(int readTimeout, int writeTimeout);
};
