%module exploresdk

%include "pyabc.i"
%include "std_vector.i"
%include "std_string.i"
%include "cpointer.i"
%include "typemaps.i"
%include "cstring.i"
%include "windows.i"
%include "exception.i"

%exception {
    try {
        $action
    } catch(const ExploreException& e) {
        SWIG_exception(SWIG_ValueError, e.what());
    }
    catch(const ExploreReadBufferException& e) {
        SWIG_exception(SWIG_MemoryError, e.what());
    }
	catch(const ExploreIOException& e) {
        SWIG_exception(SWIG_IOError, e.what());
    }
    catch(const ExploreNoBluetoothException& e) {
        SWIG_exception(SWIG_SystemError, e.what());
    }
    catch(const ExploreBtSocketException& e) {
        SWIG_exception(SWIG_TypeError, e.what());
    }
    catch(const std::exception& e) {
        SWIG_exception(SWIG_UnknownError, "Standard exception");
    } catch(...) {
        SWIG_exception(SWIG_RuntimeError, "Unknown exception");
    }
}
%include serialport.i
%include exploresearch.i



