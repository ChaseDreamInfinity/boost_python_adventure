// hello_boost_python.cpp : Defines the exported functions for the DLL application.
//

#include <vector>


char const* greet()
{
	return "hello, world";
}

char const* hello(char* str)
{
	return str;
}

int summation(int i, int j)
{
	return i + j;
}


#include <boost/python.hpp>

// Converts a C++ vector to a python list
template <class T>
boost::python::list toPythonList(std::vector<T> vector) 
{
	typename std::vector<T>::iterator iter;
	boost::python::list list;
	for (iter = vector.begin(); iter != vector.end(); ++iter) 
	{
		list.append(*iter);
	}
	return list;
}

boost::python::list vec_to_list()
{
	std::vector<int> v{1,2,3,4};
	return toPythonList(v);
}


BOOST_PYTHON_MODULE(hello_boost_python)
{
	using namespace boost::python;
	def("greet", greet);
	def("hello", hello);
	def("sum", summation);
	def("vec_to_list", vec_to_list);
}
