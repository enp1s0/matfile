#ifndef __MATFILE_HPP__
#define __MATFILE_HPP__
#include <fstream>
#include <stdexcept>

namespace mtk {
namespace matfile {
namespace detail {
struct file_header {
	enum data_t {
		fp32,
		fp64
	} data_type;
	enum matrix_t {
		dense
	} matrix_type;
	std::uint64_t m;
	std::uint64_t n;
	// For the future use
	std::uint64_t a0;
	std::uint64_t a1;
	std::uint64_t a2;
	std::uint64_t a3;
};

template <class T>
inline file_header::data_t get_data_type();
template <> inline file_header::data_t get_data_type<double>() {return file_header::fp64;};
template <> inline file_header::data_t get_data_type<float >() {return file_header::fp32;};

template <class T>
inline std::string get_type_name_str();
template <> inline std::string get_type_name_str<double>() {return "double";}
template <> inline std::string get_type_name_str<float >() {return "float" ;}

inline std::string get_data_type_str(const file_header::data_t data_t) {
	if (data_t == file_header::data_t::fp32) return "float";
	else return "double";
}
} // namespace detail

template <class T>
void save_dense(
		const std::uint64_t m,
		const std::uint64_t n,
		const T* const mat_ptr,
		const std::uint64_t ld,
		const std::string mat_name
		) {
	detail::file_header file_header;
	file_header.data_type = detail::get_data_type<T>();
	file_header.m = m;
	file_header.n = n;
	file_header.matrix_type = detail::file_header::dense;

	std::ofstream ofs(mat_name, std::ios::binary);
	ofs.write(reinterpret_cast<char*>(&file_header), sizeof(file_header));

	for (std::uint64_t j = 0; j < n; j++) {
		for (std::uint64_t i = 0; i < m; i++) {
			const auto index = i + j * ld;
			const auto v = mat_ptr[index];
			ofs.write(reinterpret_cast<const char*>(&v), sizeof(T));
		}
	}
}

template <class T>
void load_size(
		std::uint64_t& m,
		std::uint64_t& n,
		const std::string mat_name
		) {
	std::ifstream ifs(mat_name, std::ios::binary);
	if (!ifs) {
		throw std::runtime_error("[matfile error] No such file : " + mat_name);
	}

	detail::file_header file_header;
	ifs.read(reinterpret_cast<char*>(&file_header), sizeof(file_header));

	m = file_header.m;
	n = file_header.n;
}

template <class T>
detail::file_header load_header(
		const std::string mat_name
		) {
	std::ifstream ifs(mat_name, std::ios::binary);
	if (!ifs) {
		throw std::runtime_error("[matfile error] No such file : " + mat_name);
	}

	detail::file_header file_header;
	ifs.read(reinterpret_cast<char*>(&file_header), sizeof(file_header));

	return file_header;
}

template <class T>
void load_dense(
		T* const mat_ptr,
		const std::uint64_t ld,
		const std::string mat_name
		) {
	std::ifstream ifs(mat_name, std::ios::binary);
	if (!ifs) {
		throw std::runtime_error("[matfile error] No such file : " + mat_name);
	}

	detail::file_header file_header;
	ifs.read(reinterpret_cast<char*>(&file_header), sizeof(file_header));

	if (file_header.data_type != detail::get_data_type<T>()) {
		throw std::runtime_error("[matfile error] matrix type is mismatch : data_t = " + detail::get_data_type_str(file_header.data_type) + ", T = " + detail::get_type_name_str<T>());
	}
	const std::uint64_t m = file_header.m;
	const std::uint64_t n = file_header.n;

	for (std::uint64_t j = 0; j < n; j++) {
		for (std::uint64_t i = 0; i < m; i++) {
			const auto index = i + j * ld;
			T v;
			ifs.read(reinterpret_cast<char*>(&v), sizeof(T));
			mat_ptr[index] = v;
		}
	}
}
} // namespace matfile
} // namespace mtk
#endif
