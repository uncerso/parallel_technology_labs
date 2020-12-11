#include "matrixes.hpp"
#include <cassert>
#include <stdexcept>
#include <iomanip>
#include <ostream>

namespace {

template <class T>
struct MatrixViewBase {
    T & mt;
    size_t y_base;
    size_t x_base;
    size_t n;

    MatrixViewBase(T & mt, size_t y_base = 0, size_t x_base = 0, size_t n = 0)
        : mt(mt)
        , y_base(y_base)
        , x_base(x_base)
        , n(n ? n : mt.n)
    {}

    double & operator()(size_t y, size_t x)      noexcept { return mt(y_base + y, x_base + x); }
    double operator() (size_t y, size_t x) const noexcept { return mt(y_base + y, x_base + x); }

    MatrixViewBase<T> GetSubblock(bool is_right, bool is_down) const noexcept {
        auto new_size = n/2;
        auto x_shift = (is_right ? new_size : 0);
        auto y_shift = (is_down ? new_size : 0);
        return MatrixViewBase(mt, y_base + y_shift, x_base + x_shift, new_size);
    }
};

using MatrixView = MatrixViewBase<RegularMatrix>;
using CMatrixView = MatrixViewBase<const RegularMatrix>;

template<class A, class B, class C>
void Multiply(A const & lhs, B const & rhs, C & res) {
    auto n = lhs.n;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < n; ++k)
                res(i,j) += lhs(i,k) * rhs(k,j);
}

struct MatrixViewPack {
    CMatrixView a;
    CMatrixView b;
    MatrixView  c;
    MatrixViewPack(CMatrixView const & a, CMatrixView const & b, MatrixView & c, bool is_right, bool is_down)
        : a(a.GetSubblock(is_right, is_down))
        , b(b.GetSubblock(is_right, is_down))
        , c(c.GetSubblock(is_right, is_down))
    {}
};


void RecursiveMultiply(CMatrixView a, CMatrixView b, MatrixView c) {
    if (a.n < 100 || (a.n & 1)) {
        Multiply(a, b, c);
        return;
    }
    auto n = a.n / 2;
    auto y = a.y_base;
    auto x = a.x_base;
    MatrixViewPack mp11(a, b, c, false, false);
    MatrixViewPack mp12(a, b, c,  true, false);
    MatrixViewPack mp21(a, b, c, false,  true);
    MatrixViewPack mp22(a, b, c,  true,  true);

    RecursiveMultiply(mp11.a, mp11.b, mp11.c);
    RecursiveMultiply(mp12.a, mp21.b, mp11.c);

    RecursiveMultiply(mp11.a, mp12.b, mp12.c);
    RecursiveMultiply(mp12.a, mp22.b, mp12.c);

    RecursiveMultiply(mp21.a, mp11.b, mp21.c);
    RecursiveMultiply(mp22.a, mp21.b, mp21.c);

    RecursiveMultiply(mp21.a, mp12.b, mp22.c);
    RecursiveMultiply(mp22.a, mp22.b, mp22.c);
}

} //namespace

void RegularMatrix::RecursiveAddProductOf(RegularMatrix const & lhs, RegularMatrix const & rhs) {
    assert(lhs.n == rhs.n);
    assert(lhs.n == n);
    RecursiveMultiply(lhs, rhs, *this);
}

void RegularMatrix::AddProductOf(RegularMatrix const & lhs, RegularMatrix const & rhs) {
    assert(lhs.n == rhs.n);
    assert(lhs.n == n);
    Multiply(lhs, rhs, *this);
}

BlockMatrix::BlockMatrix(size_t const blocks, size_t const block_size) 
    : Base(blocks, RowBase(blocks, RegularMatrix(block_size)))
    , block_size(block_size)
{}

BlockMatrix::BlockMatrix(RegularMatrix const & mt, size_t const block_size) 
    : block_size(block_size)
{
    const auto blocks = mt.n / block_size;
    if (blocks * block_size != mt.n)
        throw std::invalid_argument("block_size should be multiple of mt.n");
    this->resize(blocks, RowBase(blocks, RegularMatrix(block_size)));
    for (size_t y = 0; y < blocks; ++y) {
        for (size_t x = 0; x < blocks; ++x) {
            auto & block = (*this)[y][x];
            for (size_t i = 0; i < block_size; ++i) {
                for (size_t j = 0; j < block_size; ++j) {
                    block(i, j) = mt(y * block_size + i, x * block_size + j);
                }
            }
        }
    }
}

std::ostream & operator<<(std::ostream & out, RegularMatrix const & mt) {
    for (size_t i = 0; i < mt.n; ++i) {
        for (size_t j = 0; j < mt.n; ++j)
            out << std::setw(4) << mt(i, j) << ' ';
        out << '\n';
    }
    return out;
}

std::ostream & operator<<(std::ostream & out, BlockMatrix const & mt) {
    constexpr size_t w = 4;
    auto n = mt.block_size;
    auto line = std::string(1 + mt[0].size() * (2 + (w + 1) * n),'-');
    out << line << '\n';
    for (auto const & block_row : mt) {
        for (size_t i = 0; i < n; ++i) {
            out << "| ";
            for (auto const & block : block_row) {
                for (size_t j = 0; j < n; ++j)
                    out << std::setw(w) << block(i, j) << ' ';
                out << "| ";
            }
            out << '\n';
        }
        out << line << '\n';
    }
    return out;
}

