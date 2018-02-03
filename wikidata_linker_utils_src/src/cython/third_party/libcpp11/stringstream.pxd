from libcpp.string cimport string

cdef extern from "<sstream>" namespace "std" nogil:
    cdef cppclass stringstream:
        stringstream& push "operator<<" (bint val);
        stringstream& push "operator<<" (short val);
        stringstream& push "operator<<" (unsigned short val);
        stringstream& push "operator<<" (int val);
        stringstream& push "operator<<" (unsigned int val);
        stringstream& push "operator<<" (long val);
        stringstream& push "operator<<" (unsigned long val);
        stringstream& push "operator<<" (float val);
        stringstream& push "operator<<" (double val);
        stringstream& push "operator<<" (long double val);
        stringstream& push "operator<<" (void* val);
        stringstream()
        string to_string "str" () const
