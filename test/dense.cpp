#include <iostream>
#include <memory>
#include <random>
#include <limits>
#include <matfile/matfile.hpp>

template <class T>
int standard_test(const std::uint64_t m, const std::uint64_t n, const std::uint64_t ld) {
	const std::string file_name = "dense_test.matrix";
	std::unique_ptr<T[]> mat(new T[ld * n]);

	std::uniform_real_distribution<double> dist(-10, 10);
	std::mt19937 mt(std::random_device{}());

	// initialize
	for (std::uint64_t i = 0; i < ld * n; i++) {
		mat.get()[i] = dist(mt);
	}

	// save to a file
	mtk::matfile::save_dense(
		m, n,
		mat.get(), ld,
		file_name
		);

	// load meta data
	const auto [load_m, load_n] = mtk::matfile::load_matrix_size(file_name);
	const auto dtype = mtk::matfile::load_dtype(file_name);
	std::printf("TEST >> shape = (%lu, %lu), ld = %lu, dtype = %s (%lu)\n",
							load_m, load_n, ld,
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
	for (std::uint64_t i = 0; i < m; i++) {
		for (std::uint64_t j = 0; j < n; j++) {
			error = std::max(std::abs(static_cast<double>(mat.get()[i + ld * j] - load_mat.get()[i + m * j])), error);
		}
	}

	const double error_threshold = std::abs(static_cast<double>(std::numeric_limits<T>::epsilon()));
	if (
		(m == load_m) && (n == load_n) && (error <= error_threshold)
		) {
		std::printf("<< PASSED\n");
		return 0;
	} else {
		std::printf("<< FAILED. The error (%e) is larger than %e\n", error, error_threshold);
		return 1;
	}
}

int main() {
	unsigned num_failed = 0;
	unsigned num_tested = 0;
	for (const auto m : std::vector<std::uint64_t>{10, 100, 1000}) {
		for (const auto n : std::vector<std::uint64_t>{10, 100, 1000}) {
			for (const auto ld_d  : std::vector<std::uint64_t>{0, 10, 100, 1000}) {
				const auto ld = m + ld_d;

				num_failed += standard_test<long double  >(m, n, ld); num_tested++;
				num_failed += standard_test<double       >(m, n, ld); num_tested++;
				num_failed += standard_test<float        >(m, n, ld); num_tested++;
				num_failed += standard_test<std::uint8_t >(m, n, ld); num_tested++;
				num_failed += standard_test<std::uint16_t>(m, n, ld); num_tested++;
				num_failed += standard_test<std::uint32_t>(m, n, ld); num_tested++;
				num_failed += standard_test<std::uint64_t>(m, n, ld); num_tested++;
				num_failed += standard_test<std::int8_t  >(m, n, ld); num_tested++;
				num_failed += standard_test<std::int16_t >(m, n, ld); num_tested++;
				num_failed += standard_test<std::int32_t >(m, n, ld); num_tested++;
				num_failed += standard_test<std::int64_t >(m, n, ld); num_tested++;
			}
		}
	}

	std::printf("[TEST RESULT] %5u / %5u PASSED\n", (num_tested - num_failed), num_tested);
}
