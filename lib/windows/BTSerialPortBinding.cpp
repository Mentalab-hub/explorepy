#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#define _WINSOCK_DEPRECATED_NO_WARNINGS

#include <winsock2.h>
#include <windows.h>
#include <ws2bth.h>
#include <string>
#include <stdlib.h>
#include "ExploreException.h"
#include "BTSerialPortBinding.h"
#include "BluetoothHelpers.h"

struct bluetooth_data
{
	SOCKET s;
	bool initialized;
};

using namespace std;

BTSerialPortBinding *BTSerialPortBinding::Create(string address, int channelID)
{
	if (channelID <= 0)
		throw ExploreException("ChannelID should be a positive int value");

	auto obj = new BTSerialPortBinding(address, channelID);

	if (!obj->data->initialized)
	{
		delete obj;
		throw ExploreException("Unable to initialize socket library");
	}

	return obj;
}

BTSerialPortBinding::BTSerialPortBinding(string address, int channelID)
	: address(address), channelID(channelID), data(new bluetooth_data())
{
	data->s = INVALID_SOCKET;
	data->initialized = BluetoothHelpers::Initialize();
}

BTSerialPortBinding::~BTSerialPortBinding()
{
	if (data->initialized)
		BluetoothHelpers::Finalize();
}

int BTSerialPortBinding::Connect()
{
	Close();
	int status = SOCKET_ERROR;

	data->s = socket(AF_BTH, SOCK_STREAM, BTHPROTO_RFCOMM);

	if (data->s != SOCKET_ERROR)
	{
		SOCKADDR_BTH addr = { 0 };
		int addrSize = sizeof(SOCKADDR_BTH);
		TCHAR addressBuffer[40];

		if (address.length() >= 40)
			throw ExploreException("Address length is invalid");

		for (size_t i = 0; i < address.length(); i++)
			addressBuffer[i] = (TCHAR)address[i];

		addressBuffer[address.length()] = 0;

		status = WSAStringToAddress(addressBuffer, AF_BTH, nullptr, (LPSOCKADDR)&addr, &addrSize);

		if (status != SOCKET_ERROR)
		{

			addr.port = channelID;
			status = connect(data->s, (LPSOCKADDR)&addr, addrSize);

			if (status != SOCKET_ERROR)
			{	
				unsigned long enableNonBlocking = 1;
				ioctlsocket(data->s, FIONBIO, &enableNonBlocking);
			}
		}
	}

	if (status != 0)
	{
		string message = BluetoothHelpers::GetWSAErrorMessage(WSAGetLastError());

		if (data->s != INVALID_SOCKET)
			closesocket(data->s);

		//throw ExploreException("Cannot connect: " + message);
	}
	return status;
}

void BTSerialPortBinding::Close()
{
	if (data->s != INVALID_SOCKET)
	{
		closesocket(data->s);
		data->s = INVALID_SOCKET;
	}
}

void BTSerialPortBinding::Read(char *buffer, int* length)
{
	if (data->s == INVALID_SOCKET)
		throw ExploreException("connection has been closed");

	if (buffer == nullptr)
		throw ExploreException("buffer cannot be null");

	if (*length == 0)
		throw ExploreException("Provided length is 0");

	fd_set set;
	FD_ZERO(&set);
	FD_SET(data->s, &set);

	int size = -1;

	//timeval timeout { 0, 0 };

	if (select(static_cast<int>(data->s) + 1, &set, nullptr, nullptr, nullptr/*&timeout*/) >= 0)
	{
		if (FD_ISSET(data->s, &set))
			size = recv(data->s, buffer, *length, 0);
		else // when no data is read from rfcomm the connection has been closed.
			size = 0; // TODO: throw ?
	}

	if (size < 0)
		throw ExploreException("Error reading from connection");

}

void BTSerialPortBinding::Write(const char *buffer, int length)
{
	if (buffer == nullptr)
		throw ExploreException("buffer cannot be null");

	if (length == 0)
		return;

	if (data->s == INVALID_SOCKET)
		throw ExploreException("Attempting to write to a closed connection");

	if (send(data->s, buffer, length, 0) != length)
		throw ExploreException("Writing attempt was unsuccessful");
}

bool BTSerialPortBinding::IsDataAvailable()
{
	if (data->s == INVALID_SOCKET)
		throw ExploreException("connection has been closed");

	u_long count;
	ioctlsocket(data->s, FIONREAD, &count);
	return count > 0;
}