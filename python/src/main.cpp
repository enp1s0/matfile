#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <matfile/matfile.hpp>
#include <cstdint>
#include <string>

template <class T>
void save_dense(
	pybind11::array_t<T, pybind11::array::f_style | pybind11::array::forcecast> mat,
	const std::string file_name
	) {
	pybind11::buffer_info buf = mat.request();

	std::size_t m, n;
	if (buf.ndim == 1) {
		m = buf.shape[0];
		n = 1;
	} else if (buf.ndim == 2){
		m = buf.shape[0];
		n = buf.shape[1];
	} else {
		throw std::runtime_error("ndim must be smaller than 3 but " + std::to_string(buf.ndim) + "is given.");
	}
	mtk::matfile::save_dense<T>(
		m, n, static_cast<T*>(buf.ptr), m, file_name
		);
}

template <class T>
pybind11::array_t<T, pybind11::array::f_style | pybind11::array::forcecast> load_dense(
	const std::string file_name
	) {

	std::size_t m, n;
	mtk::matfile::load_matrix_size(m, n, file_name);

	T* ptr = new T[m * n];
	mtk::matfile::load_dense(ptr, m, file_name);

	pybind11::capsule destroy(ptr, [](void *f) {
		T *p = reinterpret_cast<T*>(f);
		delete [] p;
	});

	if (n == 1) {
		return pybind11::array_t<T, pybind11::array::f_style | pybind11::array::forcecast>(
			{m},
			{sizeof(T)},
			ptr,
			destroy
			);
	} else {
		return pybind11::array_t<T, pybind11::array::f_style | pybind11::array::forcecast>(
			{m, n},
			{sizeof(T), m * sizeof(T)},
			ptr,
			destroy
			);
	}
}

unsigned get_fp_bit(const std::string file_name) {
	const auto info = mtk::matfile::load_header(file_name);

	if (info.data_type == mtk::matfile::data_t::fp32) {
		return 32;
	} else {
		return 64;
	}
}

PYBIND11_MODULE(matfile, m) {
    m.doc() = "matfile";

    m.def("save_dense_fp32"    , &save_dense<float >, "save_dense_fp32", pybind11::arg("matrix"), pybind11::arg("file_name"));
    m.def("save_dense_fp64"    , &save_dense<double>, "save_dense_fp64", pybind11::arg("matrix"), pybind11::arg("file_name"));
    m.def("load_dense_fp32"    , &load_dense<float >, "load_dense_fp32", pybind11::arg("file_name"));
    m.def("load_dense_fp64"    , &load_dense<double>, "load_dense_fp64", pybind11::arg("file_name"));
    m.def("get_fp_bit"         , &get_fp_bit        , "get_fp_bit"     , pybind11::arg("file_name"));
}

