#include "matrixes.hpp"
#include <cassert>
#include <stdexcept>

void RegularMatrix::AddProductOf(RegularMatrix const & lhs, RegularMatrix const & rhs) {
    assert(lhs.n == rhs.n);
    assert(lhs.n == n);
    auto & res = *this;

    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < n; ++k)
                res(i,j) += lhs(i,k) * rhs(k,j);
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

