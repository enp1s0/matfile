#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <matfile/matfile.hpp>
#include <cstdint>
#include <variant>
#include <string>

template <class T>
void save_dense(
	pybind11::array_t<T> mat,
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
		m, n, static_cast<T*>(buf.ptr), n, file_name, mtk::matfile::op_t::transpose
		);
}

template <class T>
using array_t = pybind11::array_t<T, pybind11::array::f_style | pybind11::array::forcecast>;

template <class T>
array_t<T> load_dense_core(const std::string filename) {
  std::size_t m, n;
  mtk::matfile::load_matrix_size(m, n, filename);
	T* ptr = new T[m * n];
	mtk::matfile::load_dense(ptr, m, filename);

	pybind11::capsule destroy(ptr, [](void *f) {
		T *p = reinterpret_cast<T*>(f);
		delete [] p;
	});

	if (n == 1) {
		return array_t<T>(
			{m},
			{sizeof(T)},
			ptr,
			destroy
			);
	} else {
		return array_t<T>(
			{m, n},
			{sizeof(T), m * sizeof(T)},
			ptr,
			destroy
			);
	}
}

pybind11::object load_dense(
	const std::string filename
	) {
	const auto info = mtk::matfile::load_header(filename);
  switch (info.data_type) {
    case mtk::matfile::data_t::fp32  : return {load_dense_core<float        >(filename)};
    case mtk::matfile::data_t::fp64  : return {load_dense_core<double       >(filename)};
    case mtk::matfile::data_t::fp128 : return {load_dense_core<long double  >(filename)};
    case mtk::matfile::data_t::int8  : return {load_dense_core<std::int8_t  >(filename)};
    case mtk::matfile::data_t::int16 : return {load_dense_core<std::int16_t >(filename)};
    case mtk::matfile::data_t::int32 : return {load_dense_core<std::int32_t >(filename)};
    case mtk::matfile::data_t::int64 : return {load_dense_core<std::int64_t >(filename)};
    case mtk::matfile::data_t::uint8 : return {load_dense_core<std::uint8_t >(filename)};
    case mtk::matfile::data_t::uint16: return {load_dense_core<std::uint16_t>(filename)};
    case mtk::matfile::data_t::uint32: return {load_dense_core<std::uint32_t>(filename)};
    case mtk::matfile::data_t::uint64: return {load_dense_core<std::uint64_t>(filename)};
    default: break;
  }
  return array_t<float>{};
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

    m.def("save_dense"    , &save_dense<std::int8_t  >, "", pybind11::arg("matrix"), pybind11::arg("file_name"));
    m.def("save_dense"    , &save_dense<std::int16_t >, "", pybind11::arg("matrix"), pybind11::arg("file_name"));
    m.def("save_dense"    , &save_dense<std::int32_t >, "", pybind11::arg("matrix"), pybind11::arg("file_name"));
    m.def("save_dense"    , &save_dense<std::int64_t >, "", pybind11::arg("matrix"), pybind11::arg("file_name"));
    m.def("save_dense"    , &save_dense<std::uint8_t >, "", pybind11::arg("matrix"), pybind11::arg("file_name"));
    m.def("save_dense"    , &save_dense<std::uint16_t>, "", pybind11::arg("matrix"), pybind11::arg("file_name"));
    m.def("save_dense"    , &save_dense<std::uint32_t>, "", pybind11::arg("matrix"), pybind11::arg("file_name"));
    m.def("save_dense"    , &save_dense<std::uint64_t>, "", pybind11::arg("matrix"), pybind11::arg("file_name"));
    m.def("save_dense"    , &save_dense<float        >, "", pybind11::arg("matrix"), pybind11::arg("file_name"));
    m.def("save_dense"    , &save_dense<double       >, "", pybind11::arg("matrix"), pybind11::arg("file_name"));
    m.def("load_dense"    , &load_dense, "", pybind11::arg("file_name"));
    //m.def("get_fp_bit"         , &get_fp_bit        , "get_fp_bit"     , pybind11::arg("file_name"));
}

