#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <string_view>
#include <type_traits>

// All utilities live in the cuda_learning namespace to avoid polluting global scope.
namespace cuda_learning {

// ---------------------------------------------------------------------------
// Data Initialization
// ---------------------------------------------------------------------------

/// Fill with sequential values: start, start+step, start+2*step, ...
template <typename T>
void fill_range(T* data, std::size_t n, T start = T{0}, T step = T{1}) {
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = start + static_cast<T>(i) * step;
    }
}

/// Fill with uniform random values in [low, high].
template <typename T>
void fill_uniform(T* data, std::size_t n, T low = T{0}, T high = T{1},
                  unsigned seed = 42) {
    std::mt19937 gen(seed);
    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(low, high);
        for (std::size_t i = 0; i < n; ++i) data[i] = dist(gen);
    } else {
        std::uniform_int_distribution<T> dist(low, high);
        for (std::size_t i = 0; i < n; ++i) data[i] = dist(gen);
    }
}

/// Fill with values drawn from N(mean, std_dev).
template <typename T>
void fill_normal(T* data, std::size_t n, T mean = T{0}, T std_dev = T{1},
                 unsigned seed = 42) {
    static_assert(std::is_floating_point_v<T>,
                  "fill_normal requires a floating-point type");
    std::mt19937 gen(seed);
    std::normal_distribution<T> dist(mean, std_dev);
    for (std::size_t i = 0; i < n; ++i) data[i] = dist(gen);
}

/// Fill every element with a constant value.
template <typename T>
void fill_constant(T* data, std::size_t n, T val) {
    std::fill(data, data + n, val);
}

// ---------------------------------------------------------------------------
// Error Metrics
// ---------------------------------------------------------------------------

/// Maximum absolute element-wise error: max_i |ref[i] - test[i]|
template <typename T>
T max_abs_error(const T* ref, const T* test, std::size_t n) {
    T max_err{0};
    for (std::size_t i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::abs(ref[i] - test[i]));
    }
    return max_err;
}

/// Mean absolute error: (1/n) * sum_i |ref[i] - test[i]|
template <typename T>
double mean_abs_error(const T* ref, const T* test, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += std::abs(static_cast<double>(ref[i]) - static_cast<double>(test[i]));
    }
    return sum / static_cast<double>(n);
}

// ---------------------------------------------------------------------------
// Correctness Check
// ---------------------------------------------------------------------------

/// Returns true if all |ref[i] - test[i]| <= atol.
/// On the first failure, prints a diagnostic if verbose == true.
template <typename T>
bool check_result(const T* ref, const T* test, std::size_t n,
                  T atol = static_cast<T>(1e-4), bool verbose = true) {
    for (std::size_t i = 0; i < n; ++i) {
        const T err = std::abs(ref[i] - test[i]);
        if (err > atol) {
            if (verbose) {
                std::cout << "[MISMATCH] idx=" << i << "  ref=" << ref[i]
                          << "  test=" << test[i] << "  err=" << err << "\n";
            }
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Pretty Printers
// ---------------------------------------------------------------------------

/// Print a 1-D array to stdout with an optional label.
template <typename T>
void print_1d(const T* data, std::size_t n, std::string_view label = "") {
    if (!label.empty()) std::cout << label << ": ";
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << data[i];
        if (i + 1 < n) std::cout << ", ";
    }
    std::cout << "\n";
}

/// Print a 2-D row-major matrix to stdout with an optional label.
template <typename T>
void print_2d(const T* data, std::size_t rows, std::size_t cols,
              std::string_view label = "") {
    if (!label.empty()) std::cout << label << ":\n";
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            std::cout << data[r * cols + c];
            if (c + 1 < cols) std::cout << "\t";
        }
        std::cout << "\n";
    }
}

}  // namespace cuda_learning