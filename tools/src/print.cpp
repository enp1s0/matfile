#include <iostream>
#include <memory>
#include <matfile/matfile.hpp>

void print_hex(const long double v) {std::printf("%llx", *reinterpret_cast<const unsigned long long*>(&v));}
void print_hex(const double      v) {std::printf("%lx" , *reinterpret_cast<const std::uint64_t*>(&v));}
void print_hex(const float       v) {std::printf("%x"  , *reinterpret_cast<const std::uint32_t*>(&v));}

template <class T>
void print_matfile(
	const std::string matfile_path,
	const bool print_hex_flag
	) {
	std::size_t m, n;
	mtk::matfile::load_matrix_size(m, n, matfile_path);
	std::unique_ptr<T> mat_uptr(new T[m * n]);

	mtk::matfile::load_dense(mat_uptr.get(), m, matfile_path);

	std::printf("# MATFILE path=%s, size=(%lu, %lu), dtype=%s\n",
							matfile_path.c_str(), m, n,
							mtk::matfile::detail::get_type_name_str<T>().c_str()
							);
	for (std::size_t i = 0; i < m; i++) {
		for (std::size_t j = 0; j < n; j++) {
			const auto v = mat_uptr.get()[i + j * m];
			if (print_hex_flag) {
				print_hex(v);
				std::printf(" ");
			} else {
				std::printf("%+.3e ", static_cast<double>(v));
			}
		}
		std::printf("\n");
	}
}

int main(int argc, char** argv) {
	if (argc <= 1) {
		std::fprintf(stderr, "%s /path/to/matfile.matrix\n", argv[0]);
		return 1;
	}
	const std::string op0 = argv[1];
	unsigned path_index = 1;
	bool print_hex_flag = false;
	if (op0 == "-hex") {
		print_hex_flag = true;
		path_index++;
	}

	for (; path_index < argc; path_index++) {
		const std::string matfile_path = argv[path_index];
		const auto matfile_header = mtk::matfile::load_header(matfile_path);

		if (matfile_header.data_type == mtk::matfile::data_t::fp32) {
			print_matfile<float>(matfile_path, print_hex_flag);
		} else if (matfile_header.data_type == mtk::matfile::data_t::fp64) {
			print_matfile<double>(matfile_path, print_hex_flag);
		} else if (matfile_header.data_type == mtk::matfile::data_t::fp128) {
			print_matfile<long double>(matfile_path, print_hex_flag);
		}
	}
}
