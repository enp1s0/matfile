#include <iostream>
#include <memory>
#include <random>
#include <limits>
#include <matfile/matfile.hpp>

template <class T>
void test(const std::uint64_t m, const std::uint64_t n) {
	const std::string file_name = "dense_test.matrix";
	std::unique_ptr<T[]> mat(new T[m * n]);

	std::uniform_real_distribution<T> dist(-1, 1);
	std::mt19937 mt(std::random_device{}());

	// initialize
	for (std::uint64_t i = 0; i < m * n; i++) {
		mat.get()[i] = dist(mt);
	}

	// save to a file
	mtk::matfile::save_dense(
		m, n,
		mat.get(), m,
		file_name
		);

	// load meta data
	const auto [load_m, load_n] = mtk::matfile::load_matrix_size(file_name);
	const auto dtype = mtk::matfile::load_dtype(file_name);
	std::printf("shape = (%lu, %lu) , dtype = %s (%lu)\n",
							load_m, load_n,
							mtk::matfile::detail::get_data_type_str(dtype).c_str(),
							mtk::matfile::get_dtype_size(dtype)
							);

	std::unique_ptr<T[]> load_mat(new T[load_m * load_n]);

	// load matrix
	mtk::matfile::load_dense(
		load_mat.get(), load_m,
		file_name
		);

	// check
	double error = 0.;
	for (std::uint64_t i = 0; i < m * n; i++) {
		error = std::max(std::abs(static_cast<double>(mat.get()[i] - load_mat.get()[i])), error);
	}

	if (
		(m == load_m) && (n == load_n) && (error < std::numeric_limits<T>::min())
		) {
		std::printf("PASSED\n");
	} else {
		std::printf("FAILED\n");
	}
}

int main() {
	test<long double>(100, 100);
	test<double>(100, 100);
	test<float>(100, 100);
}
