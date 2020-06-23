/**
 * @file ExploreSDK.h
 * @brief ExploreSDK class to search and retrieve available bluetooth device information
 *
 */

#pragma once

#include <vector>
#include <string>
#include <ctime>

// structure that holds information of a device
struct device
{
	std::string address;
	std::string name;
	std::time_t lastSeen;
	std::time_t lastUsed;
	bool connected;
	bool remembered;
	bool authenticated;
};

class ExploreSDK
{
private:
#ifdef _WINDOWS_
	bool initialized;
#endif
/**
    @brief Constructor for ExploreSDK, please do not call this directly, rather call it through the static fucntion ExploreSDK::Create

    @param
    @return
*/
	ExploreSDK();

public:
/**
    @brief    Destructor for ExploreSDK, please do not call this directly

    @param
    @return
*/
	~ExploreSDK();
/**
    Static function to initialize ExploreSDK instance

    @param none
    @return pointer to initialized ExploreSDK
*/
	static ExploreSDK *Create();

    /**
    Destructor for ExploreSDK, please do not call this directly

    @param
    @return
*/

/**
@brief Searches for available bluetooth devices nearby,

@param length the number of devices to search for, default length is 8 devices
@return vector<device> a vector containing device structure
*/
	std::vector<device> PerformDeviceSearch(int length = 8);
	int SdpSearch(std::string address);
};
