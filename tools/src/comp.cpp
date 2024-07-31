#include <matfile/matfile.hpp>
#include <iostream>
#include <memory>
#include <cmath>

template <class T, class S>
void comp(
	const std::string matrix_A_path,
	const std::string matrix_B_path,
	const std::size_t m,
	const std::size_t n
	) {
	std::unique_ptr<T[]> matrix_A(new T [m * n]);
	std::unique_ptr<S[]> matrix_B(new S [m * n]);

	mtk::matfile::load_dense(matrix_A.get(), m, matrix_A_path);
	mtk::matfile::load_dense(matrix_B.get(), m, matrix_B_path);

	const T* const matrix_A_ptr = matrix_A.get();
	const S* const matrix_B_ptr = matrix_B.get();

	long double base_norm2 = 0;
	long double diff_norm2 = 0;
	long double max_error = 0;
#pragma omp parallel for reduction(+: base_norm2) reduction(+: diff_norm2) reduction(max: max_error) collapse(2)
	for (std::size_t im = 0; im < m; im++) {
		for (std::size_t in = 0; in < n; in++) {
			const long double base = matrix_A_ptr[im + in * n];
			const long double diff = matrix_A_ptr[im + in * n] - matrix_B_ptr[im + in * m];

			base_norm2 += base * base;
			diff_norm2 += diff * diff;
			max_error = std::max(std::abs(diff), max_error);
		}
	}
	std::printf("relative residual = %e, max absolute error = %e\n",
							static_cast<double>(base_norm2 == 0 ? 1. : std::sqrt(diff_norm2 / base_norm2)),
							static_cast<double>(max_error)
							);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		std::fprintf(stderr, "Usage: %s [/path/to/matrixA] [/path/to/matrixB]\n", argv[0]);
		return 1;
	}

	const std::string matrix_A_path = argv[1];
	const std::string matrix_B_path = argv[2];

	const auto matrix_A_info = mtk::matfile::load_header(matrix_A_path);
	const auto matrix_B_info = mtk::matfile::load_header(matrix_B_path);

	if (matrix_A_info.matrix_type != matrix_B_info.matrix_type) {std::printf("The matrix types are mismatch\n"); return 1;}
	if (matrix_A_info.m != matrix_B_info.m || matrix_A_info.n != matrix_B_info.n) {std::printf("The matrix sizes are mismatch\n"); return 1;}

	if (matrix_A_info.data_type == mtk::matfile::data_t::fp32) {
		using T = float;
		if (matrix_B_info.data_type == mtk::matfile::data_t::fp32) {
			using S = float;
			comp<T, S>(matrix_A_path, matrix_B_path, matrix_A_info.m, matrix_A_info.n);
		} else if (matrix_B_info.data_type == mtk::matfile::data_t::fp64) {
			using S = double;
			comp<T, S>(matrix_A_path, matrix_B_path, matrix_A_info.m, matrix_A_info.n);
		} else if (matrix_B_info.data_type == mtk::matfile::data_t::fp128) {
			using S = long double;
			comp<T, S>(matrix_A_path, matrix_B_path, matrix_A_info.m, matrix_A_info.n);
		}
	}else if (matrix_A_info.data_type == mtk::matfile::data_t::fp64) {
		using T = double;
		if (matrix_B_info.data_type == mtk::matfile::data_t::fp32) {
			using S = float;
			comp<T, S>(matrix_A_path, matrix_B_path, matrix_A_info.m, matrix_A_info.n);
		} else if (matrix_B_info.data_type == mtk::matfile::data_t::fp64) {
			using S = double;
			comp<T, S>(matrix_A_path, matrix_B_path, matrix_A_info.m, matrix_A_info.n);
		} else if (matrix_B_info.data_type == mtk::matfile::data_t::fp128) {
			using S = long double;
			comp<T, S>(matrix_A_path, matrix_B_path, matrix_A_info.m, matrix_A_info.n);
		}
	}else if (matrix_A_info.data_type == mtk::matfile::data_t::fp128) {
		using T = long double;
		if (matrix_B_info.data_type == mtk::matfile::data_t::fp32) {
			using S = float;
			comp<T, S>(matrix_A_path, matrix_B_path, matrix_A_info.m, matrix_A_info.n);
		} else if (matrix_B_info.data_type == mtk::matfile::data_t::fp64) {
			using S = double;
			comp<T, S>(matrix_A_path, matrix_B_path, matrix_A_info.m, matrix_A_info.n);
		} else if (matrix_B_info.data_type == mtk::matfile::data_t::fp128) {
			using S = long double;
			comp<T, S>(matrix_A_path, matrix_B_path, matrix_A_info.m, matrix_A_info.n);
		}
	}
}
