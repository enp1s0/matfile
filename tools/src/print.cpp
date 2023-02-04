#include <iostream>
#include <memory>
#include <matfile/matfile.hpp>

template <class T>
void print_matfile(
	const std::string matfile_path
	) {
	std::size_t m, n;
	mtk::matfile::load_size(m, n, matfile_path);
	std::unique_ptr<T> mat_uptr(new T[m * n]);

	mtk::matfile::load_dense(mat_uptr.get(), m, matfile_path);

	for (std::size_t i = 0; i < m; i++) {
		for (std::size_t j = 0; j < n; j++) {
			const auto v = mat_uptr.get()[i + j * m];
			std::printf("%+.3e ", static_cast<double>(v));
		}
		std::printf("\n");
	}
}

int main(int argc, char** argv) {
	if (argc <= 1) {
		std::fprintf(stderr, "%s /path/to/matfile.matrix\n", argv[0]);
		return 1;
	}

	const std::string matfile_path = argv[1];
	const auto matfile_header = mtk::matfile::load_header(matfile_path);

	if (matfile_header.data_type == mtk::matfile::detail::file_header::fp32) {
		print_matfile<float>(matfile_path);
	} else if (matfile_header.data_type == mtk::matfile::detail::file_header::fp64) {
		print_matfile<double>(matfile_path);
	} else if (matfile_header.data_type == mtk::matfile::detail::file_header::fp128) {
		print_matfile<long double>(matfile_path);
	}
}
