#pragma once

#include <vector>
#include <string>
#include <ctime>

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
	ExploreSDK();

public:
	~ExploreSDK();
	static ExploreSDK *Create();
	std::vector<device> PerformDeviceSearch(int length = 8);
	int SdpSearch(std::string address);
};
