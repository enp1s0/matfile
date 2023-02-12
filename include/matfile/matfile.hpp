#ifndef __MATFILE_HPP__
#define __MATFILE_HPP__
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mtk {
namespace matfile {
enum data_t {
	fp32,
	fp64,
	fp128
};
enum matrix_t {
	dense
};

namespace detail {
struct file_header {
#ifndef MATFILE_USE_OLD_FORMAT
	std::uint32_t version;
#endif
	data_t data_type;
	matrix_t matrix_type;
	std::uint64_t m;
	std::uint64_t n;
	// For the future use
	std::uint64_t a0;
	std::uint64_t a1;
	std::uint64_t a2;
	std::uint64_t a3;
};

template <class T>
inline data_t get_data_type();
template <> inline data_t get_data_type<long double>() {return fp128;};
template <> inline data_t get_data_type<double     >() {return fp64;};
template <> inline data_t get_data_type<float      >() {return fp32;};

template <class T>
inline std::string get_type_name_str();
template <> inline std::string get_type_name_str<long double>() {return "long double";}
template <> inline std::string get_type_name_str<double     >() {return "double";}
template <> inline std::string get_type_name_str<float      >() {return "float" ;}

inline std::string get_data_type_str(const data_t data_t) {
	if (data_t == data_t::fp32) return "float";
	else if (data_t == data_t::fp64) return "double";
	else return "long double";
}

inline std::uint32_t get_version_uint32(
		const std::uint32_t major,
		const std::uint32_t minor
		) {
	return major * 1000 + minor;
}

inline std::uint32_t get_minor_version(
		const std::uint32_t version
		) {
	return version % 1000;
}

inline std::uint32_t get_major_version(
		const std::uint32_t version
		) {
	return version / 1000;
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
	file_header.matrix_type = matrix_t::dense;
#ifndef OLD_VERSION
	file_header.version = detail::get_version_uint32(0, 3);
#endif

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

template <class INT_T>
inline void load_size(
		INT_T& m,
		INT_T& n,
		const std::string mat_name
		) {
	const auto file_header = load_header(mat_name);

	m = file_header.m;
	n = file_header.n;
}

inline data_t load_dtype(
		const std::string mat_name
		) {
	const auto file_header = load_header(mat_name);

	return file_header.data_type;
}

inline std::size_t get_dtype_size(
		const data_t dtype
		) {
	switch (dtype) {
	case fp32: return 4;
	case fp64: return 8;
	case fp128: return 16;
	default: return 0;
	}
	return 0;
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
template <class T>
inline void print_matrix(
		const std::size_t m,
		const std::size_t n,
		const T* const ptr,
		const std::size_t ld,
		const std::string name = ""
		) {
	if (name.length() != 0) {
		std::printf("%s = \n", name.c_str());
	}
	for (std::size_t mi = 0; mi < m; mi++) {
		for (std::size_t ni = 0; ni < n; ni++) {
			const auto v = ptr[mi + ni * ld];
			std::printf("%+.3e ", v);
		}
		std::printf("\n");
	}
}
template <class T>
inline void print_matrix(
		const std::size_t m,
		const std::size_t n,
		const T* const ptr,
		const std::string name = ""
		) {
	print_matrix(m, n, ptr, m, name);
}
} // namespace matfile
} // namespace mtk
#endif
