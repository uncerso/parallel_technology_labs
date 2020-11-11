#include "matrix.hpp"
#include <istream>
#include <cmath>
#include <omp.h>
#include <cassert>

using namespace std;

istream & operator >> (istream & inp, Matrix & mt) {
    for (auto & row : mt)
        for (auto & v : row)
            inp >> v;
    return inp;
}

ostream & operator << (ostream & out, Matrix const & mt) {
    for (auto & row : mt) {
        for (auto & v : row)
            out << v << ' ';
        out << '\n';
    }
    return out;
}

void Matrix::transpose() noexcept {
    auto n = size();
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            std::swap((*this)[i][j], (*this)[j][i]);
}

void Matrix::make_symmetric() noexcept {
    auto n = size();
    auto & mt = *this;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            auto tmp = (mt[i][j] + mt[j][i]) / 2;
            mt[i][j] = tmp;
            mt[j][i] = tmp;
        }
    }
}


Matrix Matrix::operator * (Matrix const & o) const {
    assert(o.size() == size());
    size_t n = size();
    Matrix res(n);

    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < n; ++k)
                res[i][j] += (*this)[i][k] * o[k][j];
    return res;
}

double dot_product(Row const & a, Row const & b, size_t amount) noexcept {
    double ans = 0;
    for (size_t i = 0; i < amount; ++i)
        ans += a[i] * b[i];
    return ans;
}

void Matrix::decompose(size_t threads) noexcept {
    auto & mt = *this;
    if (size() <= 1)
        return;

    mt[0][0] = sqrt(mt[0][0]);
    for (size_t i = 1; i < size(); ++i)
        mt[i][0] = mt[i][0] / mt[0][0];

    #pragma omp parallel num_threads(threads)
    {
    for (size_t i = 1; i < size(); ++i) {
        mt[i][i] = sqrt(mt[i][i] - dot_product(mt[i], mt[i], i));
        #pragma omp for
        for (size_t j = i + 1; j < size(); ++j) {
            mt[j][i] -= dot_product(mt[i], mt[j], i);
            mt[j][i] /= mt[i][i];
        }
    }
    }
}

Row Matrix::map(Row const & b) const {
    Row ans(size());
    auto & a = *this;
    for (size_t i = 0; i < size(); ++i) {
        ans[i] = 0;
        for (size_t j = 0; j < size(); ++j)
            ans[i] += a[i][j] * b[j]; 
    }
    return ans;
}

void Matrix::back_transpose_map(Row & b) const {
    auto & a = *this;
    for (size_t i = size(); 0 < i--;) {
        for (size_t j = size(); i + 1 < j--;)
            b[i] -= a[j][i] * b[j];
        b[i] /= a[i][i];
    }
}

void Matrix::back_map(Row & b) const {
    auto & a = *this;
    for (size_t i = 0; i < size(); ++i) {
        b[i] -= dot_product(a[i], b, i);
        b[i] /= a[i][i];
    }
}

void Matrix::solve(Row & b) const {
    assert(b.size() == size());
    back_map(b);
    back_transpose_map(b);
}

bool Matrix::operator==(Matrix const & o) const noexcept {
    if (size() != o.size())
        return false;
    auto const & a = *this;
    for (size_t i = 0; i < size(); ++i) {
        for (size_t j = 0; j < size(); ++j) {
            if (0 < abs(a[i][j] - o[i][j]))
                return false;
        }
    }
    return true;
}
