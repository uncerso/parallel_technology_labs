#pragma once
#include <vector>
#include <cstddef>

struct RegularMatrix : std::vector<double> {
    using Base = std::vector<double>;
    size_t n;
    RegularMatrix(size_t n) : Base(n*n, 0.0), n(n) {}
    RegularMatrix(RegularMatrix &&) = default;
    RegularMatrix(RegularMatrix const &) = default;
    RegularMatrix& operator=(RegularMatrix &&) = default;
    RegularMatrix& operator=(RegularMatrix const &) = default;

    double & operator()(size_t y, size_t x)      noexcept { return (*this)[y * n + x]; }
    double operator() (size_t y, size_t x) const noexcept { return (*this)[y * n + x]; }

    void AddProductOf(RegularMatrix const & lhs, RegularMatrix const & rhs);
};

struct BlockMatrix : std::vector<std::vector<RegularMatrix>> {
    using RowBase = std::vector<RegularMatrix>;
    using Base = std::vector<RowBase>;
    
    size_t block_size;
    BlockMatrix(size_t const blocks, size_t const block_size);
    BlockMatrix(RegularMatrix const & mt, size_t const block_size);
};
