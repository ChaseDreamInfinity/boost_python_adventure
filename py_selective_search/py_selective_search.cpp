// SeleciveSearchPython.cpp : Defines the exported functions for the DLL application.
//
#include <vector>
#include <boost/python.hpp>

#include "selective_search.h"

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


boost::python::list get_windows(char* filename)
{
	// Generate the bounding boxes through multi-scale, multi-simfunction, multi-colormap based selective search
	Mat bbox = selective_search_rcnn(filename);

	// Convert the Mat to a C++ vector of boost::python::list 
	vector<boost::python::list> bbox_vec2d(bbox.rows);
	for (int y = 0; y < bbox.rows; y++) {
		for (int x = 0; x < bbox.cols; x++) {
			bbox_vec2d[y].append(bbox.at<int>(y, x));
		}
	}

	return toPythonList(bbox_vec2d);
}


BOOST_PYTHON_MODULE(py_selective_search)
{
	using namespace boost::python;
	def("get_windows", get_windows);
}


