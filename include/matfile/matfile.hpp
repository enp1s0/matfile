#ifndef __MATFILE_HPP__
#define __MATFILE_HPP__
#include <fstream>
#include <sstream>
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
	ofs.close();
}

template <class INT_T>
inline void load_size(
		INT_T& m,
		INT_T& n,
		const std::string mat_name
		) {
	std::ifstream ifs(mat_name, std::ios::binary);
	if (!ifs) {
		throw std::runtime_error("[matfile error] No such file : " + mat_name);
	}

	detail::file_header file_header;
	ifs.read(reinterpret_cast<char*>(&file_header), sizeof(file_header));
	ifs.close();

	m = file_header.m;
	n = file_header.n;
}

inline detail::file_header load_header(
		const std::string mat_name
		) {
	std::ifstream ifs(mat_name, std::ios::binary);
	if (!ifs) {
		throw std::runtime_error("[matfile error] No such file : " + mat_name);
	}

	detail::file_header file_header;
	ifs.read(reinterpret_cast<char*>(&file_header), sizeof(file_header));
	ifs.close();

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
	ifs.close();
}

namespace matrix_market {
namespace detail {
using matrix_type_t = unsigned;
constexpr matrix_type_t unsupported_matrix = 0;
constexpr matrix_type_t general_matrix   = 1;
constexpr matrix_type_t symmetric_matrix = 2;

using element_type_t = unsigned;
constexpr element_type_t unsupported_value = 0;
constexpr element_type_t real_value = 1;
constexpr element_type_t pattern_value = 2;

inline matrix_type_t get_matrix_type(
		const std::string banner
		) {
	if (banner.find("general") != std::string::npos) {
		return general_matrix;
	} else if (banner.find("symmetric") != std::string::npos) {
		return symmetric_matrix;
	}
	return unsupported_matrix;
}

inline element_type_t get_element_type(
		const std::string banner
		) {
	if (banner.find("real") != std::string::npos) {
		return real_value;
	} else if (banner.find("pattern") != std::string::npos) {
		return pattern_value;
	}
	return unsupported_value;
}
} // unnamed namespace

template <class INT_T>
void load_matrix_size(
		INT_T& m,
		INT_T& n,
		const std::string filepath
		) {
	std::ifstream ifs(filepath);

	if (!ifs) {
		throw std::runtime_error("[matfile error] No such file : " + filepath);
	}

	std::string line;
	while (std::getline(ifs, line)) {
		if (line[0] != '%') {
			break;
		}
	}
	std::size_t num_elements;

	std::stringstream ss(line);
	ss >> m >> n >> num_elements;

	ifs.close();
}

template <class INT_T = std::size_t>
std::pair<INT_T, INT_T> load_matrix_size(
		const std::string filepath
		) {
	std::size_t m, n;
	load_matrix_size(m, n, filepath);

	return std::pair<INT_T, INT_T>{m, n};
}

template <class T>
void load_matrix(
		T* const ptr,
		const std::size_t ld,
		const std::string filepath,
		const bool fill_zero = true
		) {
	std::ifstream ifs(filepath);

	if (!ifs) {
		throw std::runtime_error("[matfile error] No such file : " + filepath);
	}

	std::string line;
	std::getline(ifs, line);

	const auto matrix_type = detail::get_matrix_type(line);
	if (matrix_type == detail::unsupported_matrix) {
		throw std::runtime_error("Unsupported matrix type : banner = " + line);
	}
	const auto element_type = detail::get_element_type(line);
	if (matrix_type == detail::unsupported_matrix) {
		throw std::runtime_error("Unsupported element type : banner = " + line);
	}

	while (std::getline(ifs, line)) {
		if (line[0] != '%') {
			break;
		}
	}
	std::size_t m, n, num_elements;

	std::stringstream ss(line);
	ss >> m >> n >> num_elements;

	if (fill_zero) {
		for (std::size_t i = 0; i < m; i++) {
			for (std::size_t j = 0; j < n; j++) {
				ptr[i + j * ld] = 0;
			}
		}
	}

	for (std::size_t l = 0; l < num_elements; l++) {
		if (element_type == detail::real_value) {
			std::size_t i, j;
			double v;
			ifs >> i >> j >> v;
			i--;j--;

			ptr[i + j * ld] = v;

			if (matrix_type == detail::symmetric_matrix) {
				ptr[j + i * ld] = v;
			}
		} else if (element_type == detail::pattern_value) {
			std::size_t i, j;
			ifs >> i >> j;
			i--;j--;

			ptr[i + j * ld] = 1;

			if (matrix_type == detail::symmetric_matrix) {
				ptr[j + i * ld] = 1;
			}
		}
	}
	ifs.close();
}
} // namespace matrix_market
} // namespace matfile
} // namespace mtk
#endif
