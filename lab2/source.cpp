#include "matrix.hpp"
#include "benchmark.hpp"
#include <iostream>
#include <random>
#include <omp.h>
#include <cassert>

using namespace std;

std::ostream & operator << (std::ostream & out, Row const & r) {
    for (auto e : r)
        out << e << ' ';
    return out;
}

double norm(Row const & v) {
    return sqrt((v * v).sum());
}

double GetError(Row const & a, Row const & b) {
    return norm(a-b)/norm(b);
}

Row GetRandomRow(size_t n, size_t max_value = 10) {
    random_device rd;
    mt19937 next_rand(rd());

    size_t bound = max_value * 2 + 1;
    Row ans(n);
    for (auto & el : ans)
        el = static_cast<double>(next_rand() % bound) - static_cast<double>(max_value);
    return ans;
}

// generate random matrix with elements from `-max_value` to `max_value`
Matrix GetRandomMatrix(size_t n, size_t max_value = 10) {
    Matrix mt(n);
    random_device rd;
    mt19937 next_rand(rd());
    
    size_t bound = max_value * 2 + 1;

    for (auto & row : mt)
        for (auto & el : row)
            el = static_cast<double>(next_rand() % bound) - static_cast<double>(max_value);
    return mt;
}

Matrix GetPositiveSymmetricMatrix(size_t n) {
    Matrix mt(n);
    random_device rd;
    mt19937 next_rand(rd());

    // generate eigenvalues
    for (size_t i = 0; i < n; ++i)
        mt[i][i] = 1.0 + static_cast<double>(next_rand() % 16);

    // generate eigenvectors
    auto s = GetRandomMatrix(n);

    // make a positive matrix
    mt = s * mt;
    s.transpose();
    mt = mt * s;

    mt.make_symmetric();
    return mt;
}

size_t get_threads(int argc, char const *argv[]) {
    if (argc < 2)
        return 1;
    return static_cast<size_t>(std::atoi(argv[1]));
}

ostream & operator << (ostream & out, Measured_time const & t) {
    out << "Mean time: " << t.mean_time << " ms, Standard deviation: " << t.standard_deviation << " ms";
    return out;
}

bool run_test(size_t n, size_t threads) {
    constexpr double eps = 1e-8;
    auto mt = GetPositiveSymmetricMatrix(n);
    auto mt_par = mt;

    auto b = GetRandomRow(n);
    auto mapped_b = mt.map(b);

    mt.decompose();
    mt_par.decompose(threads);
    if (!(mt == mt_par)) {
        cout << "WTF\n";
        cout << mt << '\n';
        cout << mt_par << '\n';
        return false;
    }

    mt.solve(mapped_b);
    return GetError(b, mapped_b) < eps;
}

int main(int argc, char const *argv[]) {
    auto threads = get_threads(argc, argv);
    omp_set_num_threads(static_cast<int>(threads));
    size_t n;
    cin >> n;

    if (run_test(n, threads))
        cout << "Correctness test passed\n";

    auto mt = GetPositiveSymmetricMatrix(n);
    auto decompose_test = [mt = mt]() mutable { mt.decompose(); };
    auto decompose_parallel_test = [mt = mt, threads]() mutable { mt.decompose(threads); };
    auto t1 = measure(decompose_test, 15);
    auto t2 = measure(decompose_parallel_test, 60);
    cout << "Single thread:\n";
    cout << t1 << '\n';
    cout << "Parallel:\n";
    cout << t2 << '\n';
    return 0;
}