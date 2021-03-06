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

void decomposing_test(Matrix const & mt, vector<size_t> thread_list) {
    cout << "----< Started decomposing test >----\n";
    size_t step = 8;
    size_t tests = 10;
    for (auto i : thread_list) {
        auto test = [mt = mt, i]() mutable { mt.decompose(i); };
        auto t = measure(test, tests);
        tests += step;
        cout << i << " threads:\n";
        cout << t << '\n';
    }
}

void solving_test(Matrix const & mt, vector<size_t> thread_list, size_t m) {
    cout << "----< Started solving test >----\n";
    vector<Row> right_parts;
    right_parts.reserve(m);
    for (size_t i = 0; i < m; ++i)
        right_parts.push_back(GetRandomRow(mt.size()));
    size_t step = 10;
    size_t tests = 0;
    for (auto i : thread_list) {
        auto test = [&mt, right_parts, i]() mutable { mt.solve(right_parts, i); };
        tests += step;
        if (step > 0) --step;
        auto t = measure(test, tests);
        cout << i << " threads:\n";
        cout << t << '\n';
    }
}

bool run_test(Matrix mt, size_t threads) {
    constexpr double eps = 1e-7;
    auto mt_par = mt;

    auto b = GetRandomRow(mt.size());
    auto mapped_b = mt.map(b);

    mt.decompose();
    mt_par.decompose(threads);
    if (!(mt == mt_par))
        return false;

    mt.solve(mapped_b);
    return GetError(b, mapped_b) < eps;
}

vector<size_t> GenThreadList(size_t max_threads) {
    vector<size_t> ans;
    size_t a = 1, b = 1, n = 1;
    while (n <= max_threads) {
        ans.push_back(n);
        if ((n & (n-1)) == 0) {
            a = b;
            b = n;
        }
        n += a;
    }
    return ans;
}

int main(int argc, char const *argv[]) {
    auto threads = get_threads(argc, argv);
    omp_set_num_threads(static_cast<int>(threads));
    auto thread_list = GenThreadList(threads);
    size_t n, m;
    cin >> n >> m;

    auto mt = GetPositiveSymmetricMatrix(n);
    if (run_test(mt, threads))
        cout << "Correctness test passed\n";

    decomposing_test(mt, thread_list);
    mt.decompose();
    solving_test(mt, thread_list, m);
    return 0;
}