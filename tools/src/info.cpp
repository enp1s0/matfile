#include <iostream>
#include <memory>
#include <matfile/matfile.hpp>
#include <fphistogram/fphistogram.hpp>

template <class T>
void print_info(
	const std::string matfile_path
	) {
	std::size_t m, n;
	mtk::matfile::load_matrix_size(m, n, matfile_path);

	std::printf("# size  : %lu x %lu\n", m, n);
	std::printf("# dtype : %s\n", mtk::matfile::detail::get_type_name_str<T>().c_str());

	auto mat_uptr = std::unique_ptr<T[]>(new T[m * n]);
	mtk::matfile::load_dense(mat_uptr.get(), m, matfile_path);

	std::printf("# exp histogram\n");
	mtk::fphistogram::print_histogram_pm(mat_uptr.get(), m * n);
}

int main(
	int argc,
	char** argv
	) {
	if (argc <= 1) {
		std::fprintf(stderr, "Usage %s: [matfile path list]\n", argv[0]);
		return 1;
	}

	for (int i = 1; i < argc; i++) {
		const auto matfile_path = std::string(argv[i]);
		std::printf("## ---- [%d] path : %s ----\n", i, matfile_path.c_str());

		const auto dtype = mtk::matfile::load_dtype(matfile_path);
		if (dtype == mtk::matfile::data_t::fp32) {
			print_info<float >(matfile_path);
		} else if (dtype == mtk::matfile::data_t::fp64) {
			print_info<double>(matfile_path);
		}
	}
}
