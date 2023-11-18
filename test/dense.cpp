#include <iostream>
#include <memory>
#include <random>
#include <limits>
#include <matfile/matfile.hpp>

template <class T>
void test(const std::uint64_t m, const std::uint64_t n) {
	const std::string file_name = "dense_test.matrix";
	std::unique_ptr<T[]> mat(new T[m * n]);

	std::uniform_real_distribution<double> dist(-10, 10);
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

	const double error_threshold = std::abs(static_cast<double>(std::numeric_limits<T>::min()));
	if (
		(m == load_m) && (n == load_n) && (error <= error_threshold)
		) {
		std::printf("PASSED\n");
	} else {
		std::printf("FAILED. The error is larger than %e\n", error_threshold);
	}
}

int main() {
	constexpr std::size_t N = 100;
	test<long double  >(N, N);
	test<double       >(N, N);
	test<float        >(N, N);
	test<std::uint8_t >(N, N);
	test<std::uint16_t>(N, N);
	test<std::uint32_t>(N, N);
	test<std::uint64_t>(N, N);
	test<std::int8_t  >(N, N);
	test<std::int16_t >(N, N);
	test<std::int32_t >(N, N);
	test<std::int64_t >(N, N);
}
