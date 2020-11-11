#pragma once
#include <vector>
#include <valarray>
#include <iosfwd>

using Row = std::valarray<double>;
// using Row = std::vector<double>;

class Matrix : public std::vector<Row> {
    using Base = std::vector<Row>;
    void back_map(Row & b) const;
    void back_transpose_map(Row & b) const;
public:
    Matrix(size_t n) : Base(n, Row(0.0, n)) {}
    // Matrix(size_t n) : Base(n, Row(n 0)) {}

    void decompose(size_t threads = 1) noexcept;
    void solve(Row & b) const;
    void solve(vector<Row> & b, size_t threads = 1) const;

    Matrix operator * (Matrix const & o) const;
    void transpose() noexcept;
    void make_symmetric() noexcept;

    Row map(Row const & b) const;
    bool operator==(Matrix const & o) const noexcept;
};

std::istream & operator >> (std::istream & inp, Matrix & mt);
std::ostream & operator << (std::ostream & out, Matrix const & mt);