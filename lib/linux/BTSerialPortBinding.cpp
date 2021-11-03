#include <iostream>
#include <string>
#include <stdlib.h>
#include <unistd.h>
#include "ExploreException.h"
#include "BTSerialPortBinding.h"

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

#include <bluetooth/bluetooth.h>
#include <bluetooth/hci.h>
#include <bluetooth/hci_lib.h>
#include <bluetooth/sdp.h>
#include <bluetooth/sdp_lib.h>
#include <bluetooth/rfcomm.h>
}

using namespace std;

struct bluetooth_data
{
	int s;
	int rep[2];
};

BTSerialPortBinding *BTSerialPortBinding::Create(string address, int channelID)
{
	if (channelID <= 0)
		//throw ExploreException("ChannelID should be a positive int value");
		fprintf(stdout, "Cannot create pipe for reading - ");

	return new BTSerialPortBinding(address, channelID);
}

BTSerialPortBinding::BTSerialPortBinding(string address, int channelID)
	: address(address), channelID(channelID), data(new bluetooth_data())
{
	data->s = 0;
}

BTSerialPortBinding::~BTSerialPortBinding()
{
	Close();
}

int BTSerialPortBinding::Connect()
{
	Close();

	// allocate an error pipe
	if (pipe(data->rep) == -1)
	{
		string err("Cannot create pipe for reading - ");
		//throw ExploreException(err + strerror(errno));
		fprintf(stdout, "Cannot create pipe for reading - ");
	}

	int flags = fcntl(data->rep[0], F_GETFL, 0);
	fcntl(data->rep[0], F_SETFL, flags | O_NONBLOCK);

	struct sockaddr_rc addr = {
		0x00,
		{ { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 } },
		0x00
	};

	// allocate a socket
	data->s = socket(AF_BLUETOOTH, SOCK_STREAM, BTPROTO_RFCOMM);

	// set the connection parameters (who to connect to)
	addr.rc_family = AF_BLUETOOTH;
	addr.rc_channel = (uint8_t)channelID;
	str2ba(address.c_str(), &addr.rc_bdaddr);

	// connect to server
	int status = connect(data->s, (struct sockaddr *)&addr, sizeof(addr));

	int sock_flags = fcntl(data->s, F_GETFL, 0);
	fcntl(data->s, F_SETFL, sock_flags | O_NONBLOCK);

	if (status != 0)
		//throw ExploreException("Connection to bluetooth device failed!");
        return  INITIAL_CONNECTION_FAILURE_ERROR;

	return STATUS_OK;
}

void BTSerialPortBinding::Close()
{
	if (data->s != 0)
	{
		close(data->s);
		write(data->rep[1], "close", (strlen("close") + 1));
		data->s = 0;
	}
}

void BTSerialPortBinding::Read(char *bt_buffer, int* bt_length)
{
	if (data->s == 0)
		throw ExploreIOException("connection has been closed");
	//allocating space in buffer

	fd_set set;
	FD_ZERO(&set);
	FD_SET(data->s, &set);
	FD_SET(data->rep[0], &set);

	int nfds = (data->s > data->rep[0]) ? data->s : data->rep[0];
	int size = -1;

	try{

	if (pselect(nfds + 1, &set, NULL, NULL, NULL, NULL) >= 0)
	{
		if (FD_ISSET(data->s, &set)){

			size = recv(data->s, bt_buffer, *bt_length, 0);

//			cout << "length is " << *bt_length << "size is " <<  size << endl;
			if(size < 0)
			{
                		throw ExploreReadBufferException("EMPTY_BUFFER_ERROR");
                		cout << "length is " << *bt_length << "size is " <<  size << endl;
			}


		}

		else // when no data is read from rfcomm the connection has been closed.
			fprintf(stdout, " no data is read from rfcomm");
	}}

	catch (abi::__forced_unwind&) {
     throw;

    }
    catch(ExploreReadBufferException &e) {
        throw ExploreReadBufferException("EMPTY_BUFFER_ERROR");
    }




}

void BTSerialPortBinding::Write(const char *write_buffer, int length)
{
	if (write_buffer == nullptr)
		fprintf(stdout, "write_buffer cannot be null");

	if (length == 0)
		return;

	if (data->s == 0)
		//throw ExploreException("Attempting to write to a closed connection");
		fprintf(stdout, "write_buffer cannot be null");

    int write_length = -1;

    try{

    //write_length = write(data->s, write_buffer, length);

    write_length = send(data->s, write_buffer, length, 0);

	if (write_length != length)
		//throw ExploreException("Writing attempt was unsuccessful");
		fprintf(stdout, "Writing attempt was unsuccessful");
    }

     catch (abi::__forced_unwind&) { // reference to the base of a polymorphic object
      throw;// information from length_error printed

    }


}

bool BTSerialPortBinding::IsDataAvailable()
{
	int count;
	ioctl(data->s, FIONREAD, &count);
	return count > 0;
}
