%module exploresearch

%{
#include "ExploreException.h"
#include "ExploreSDK.h"

%}

%template(vectordevice) std::vector<device>;



%include "ExploreException.h"
%include "ExploreSDK.h"
