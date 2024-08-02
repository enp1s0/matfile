#ifndef __MATFILE_HPP__
#define __MATFILE_HPP__
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdint>

namespace mtk {
namespace matfile {
enum class data_t {
	int8,
	int16,
	int32,
	int64,
	uint8,
	uint16,
	uint32,
	uint64,
	fp32,
	fp64,
	fp128
};
enum class matrix_t {
	dense
};

enum class op_t {
	transpose,
	no_transpose
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
template <> inline data_t get_data_type<long double  >() {return data_t::fp128 ;};
template <> inline data_t get_data_type<double       >() {return data_t::fp64  ;};
template <> inline data_t get_data_type<float        >() {return data_t::fp32  ;};
template <> inline data_t get_data_type<std::int64_t >() {return data_t::int64 ;};
template <> inline data_t get_data_type<std::int32_t >() {return data_t::int32 ;};
template <> inline data_t get_data_type<std::int16_t >() {return data_t::int16 ;};
template <> inline data_t get_data_type<std::int8_t  >() {return data_t::int8  ;};
template <> inline data_t get_data_type<std::uint64_t>() {return data_t::uint64;};
template <> inline data_t get_data_type<std::uint32_t>() {return data_t::uint32;};
template <> inline data_t get_data_type<std::uint16_t>() {return data_t::uint16;};
template <> inline data_t get_data_type<std::uint8_t >() {return data_t::uint8 ;};

inline std::string get_data_type_str(const data_t data_t) {
	switch (data_t) {
	case data_t::fp128 : return "long double";
	case data_t::fp64  : return "double"     ;
	case data_t::fp32  : return "float"      ;
	case data_t::int64 : return "int64_t"    ;
	case data_t::int32 : return "int32_t"    ;
	case data_t::int16 : return "int16_t"    ;
	case data_t::int8  : return "int8_t"     ;
	case data_t::uint64: return "uint64_t"   ;
	case data_t::uint32: return "uint32_t"   ;
	case data_t::uint16: return "uint16_t"   ;
	case data_t::uint8 : return "uint8_t"    ;
	default:
		break;
	}
	return "Unknown";
}

template <class T>
inline std::string get_type_name_str() {
	return get_data_type_str(get_data_type<T>());
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
inline void load_matrix_size(
		INT_T& m,
		INT_T& n,
		const std::string mat_name
		) {
	const auto file_header = load_header(mat_name);

	m = file_header.m;
	n = file_header.n;
}

template <class INT_T = std::size_t>
std::pair<INT_T, INT_T> load_matrix_size(
		const std::string filepath
		) {
	std::size_t m, n;
	load_matrix_size(m, n, filepath);

	return std::pair<INT_T, INT_T>{m, n};
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
	case data_t::int8: return 1;
	case data_t::int16: return 2;
	case data_t::int32: return 4;
	case data_t::int64: return 8;
	case data_t::uint8: return 1;
	case data_t::uint16: return 2;
	case data_t::uint32: return 4;
	case data_t::uint64: return 8;
	case data_t::fp32: return 4;
	case data_t::fp64: return 8;
	case data_t::fp128: return 16;
	default: break;
	}
	return 0;
}

namespace detail {
template <class T, class MATFILE_T>
void load_dense_core(
		T* const ptr,
		std::ifstream& ifs,
		const std::size_t m,
		const std::size_t n,
		const std::uint64_t ld,
		const op_t op
		) {
	for (std::uint64_t j = 0; j < n; j++) {
		for (std::uint64_t i = 0; i < m; i++) {
			std::size_t index;
			if (op == op_t::no_transpose) {
				index = i + j * ld;
			} else {
				index = j + i * ld;
			}
			MATFILE_T v;
			ifs.read(reinterpret_cast<char*>(&v), sizeof(T));
			ptr[index] = v;
		}
	}
}

} // namespace detail

template <class T>
void load_dense(
		T* const mat_ptr,
		const std::uint64_t ld,
		const std::string mat_name,
		const op_t op = op_t::no_transpose
		) {
	std::ifstream ifs(mat_name, std::ios::binary);
	if (!ifs) {
		throw std::runtime_error("[matfile error] No such file : " + mat_name);
	}

	detail::file_header file_header;
	ifs.read(reinterpret_cast<char*>(&file_header), sizeof(file_header));

	const std::uint64_t m = file_header.m;
	const std::uint64_t n = file_header.n;
	const auto dtype = file_header.data_type;

	switch (dtype) {
#define LOAD_DENSE_CODE(MATFILE_T, data_type) \
	case data_t::data_type: \
		detail::load_dense_core<T, MATFILE_T>(mat_ptr, ifs, m, n, ld, op); \
		break
		LOAD_DENSE_CODE(long double, fp128);
		LOAD_DENSE_CODE(double, fp64);
		LOAD_DENSE_CODE(float, fp32);
		LOAD_DENSE_CODE(std::uint8_t , uint8);
		LOAD_DENSE_CODE(std::uint16_t, uint16);
		LOAD_DENSE_CODE(std::uint32_t, uint32);
		LOAD_DENSE_CODE(std::uint64_t, uint64);
		LOAD_DENSE_CODE(std::int8_t , int8);
		LOAD_DENSE_CODE(std::int16_t, int16);
		LOAD_DENSE_CODE(std::int32_t, int32);
		LOAD_DENSE_CODE(std::int64_t, int64);
	default:
		break;
	}

	ifs.close();
}

template <class T, class MATFILE_T = T>
void save_dense(
		const std::uint64_t m,
		const std::uint64_t n,
		const T* const mat_ptr,
		const std::uint64_t ld,
		const std::string mat_name,
		const op_t op = op_t::no_transpose
		) {
	detail::file_header file_header;
	file_header.data_type = detail::get_data_type<MATFILE_T>();
	file_header.m = m;
	file_header.n = n;
	file_header.matrix_type = matrix_t::dense;
#ifndef MATFILE_USE_OLD_FORMAT
	file_header.version = detail::get_version_uint32(0, 7);
#endif

	std::ofstream ofs(mat_name, std::ios::binary);
	ofs.write(reinterpret_cast<char*>(&file_header), sizeof(file_header));

	for (std::uint64_t j = 0; j < n; j++) {
		for (std::uint64_t i = 0; i < m; i++) {
			std::size_t index;
			if (op == op_t::no_transpose) {
				index = i + j * ld;
			} else {
				index = j + i * ld;
			}
			const MATFILE_T v = mat_ptr[index];
			ofs.write(reinterpret_cast<const char*>(&v), sizeof(MATFILE_T));
		}
	}
	ofs.close();
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

namespace detail {
template <class INT_T>
struct
load_size_proxy {
[[deprecated("Reason: `load_size` is deprecated. Please use load_matrix_size instead.")]]
  void operator()(
		INT_T& m,
		INT_T& n,
		const std::string mat_name
      ) {
    load_matrix_size(m, n, mat_name);
  }
[[deprecated("Reason: `load_size` is deprecated. Please use load_matrix_size instead.")]]
  std::pair<INT_T, INT_T> operator()(
		const std::string mat_name
      ) {
    return load_matrix_size(mat_name);
  }
};
} // namespace detail

template <class INT_T>
inline void load_size(
		INT_T& m,
		INT_T& n,
		const std::string mat_name
		) {
  detail::load_size_proxy<INT_T>{}(m, n, mat_name);
}

template <class INT_T = std::uint64_t>
inline std::pair<INT_T, INT_T> load_size(
		const std::string mat_name
		) {
  return detail::load_size_proxy<INT_T>{}(mat_name);
}
} // namespace matfile
} // namespace mtk
#endif
