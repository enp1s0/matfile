#include <memory>
#include <cmath>
#include <iostream>
#include <matfile/matfile.hpp>

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "Usage : %s /path/to/matrix.mtx\n", argv[0]);
		return 1;
	}

	const auto [m, n] = mtk::matfile::matrix_market::load_matrix_size(argv[1]);

	std::unique_ptr<float[]> mat_uptr(new float[m * n]);

	mtk::matfile::matrix_market::load_matrix(mat_uptr.get(), m, argv[1]);

	double sum = 0;
	for (std::size_t i = 0; i < m; i++) {
		for (std::size_t j = 0; j < n; j++) {
			const auto v = mat_uptr.get()[i + m * j];

			sum += v * v;
		}
	}

	std::printf("File : %s, shape = (%lu, %lu) l2 norm = %e\n",
							argv[1],
							m, n,
							std::sqrt(sum)
							);
}
