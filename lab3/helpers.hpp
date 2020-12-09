#pragma once
#include "matrixes.hpp"
#include <vector>
#include <memory>
#include <mpi.h>

struct InitGuard {
    InitGuard(int argc, char **argv) {
        MPI::Init(argc, argv);
    }

    ~InitGuard() {
        MPI::Finalize();
    }
};

struct ProcInfo {
    int rank = 0;
    size_t x = 0;
    size_t y = 0;
};

struct Comms {
    size_t grid_size;
    MPI::Cartcomm grid;
    MPI::Cartcomm col;
    MPI::Cartcomm row;
};

struct State {
    RegularMatrix persistent_a;
    RegularMatrix a, c;
    std::unique_ptr<RegularMatrix> b1, b2;
    State(size_t n) 
        : persistent_a(n)
        , a(n)
        , c(n)
        , b1(std::make_unique<RegularMatrix>(n))
        , b2(std::make_unique<RegularMatrix>(n))
    {}
};

class Win {
    MPI::Win win;
    MPI::Cartcomm const & comm;
public:
    Win(RegularMatrix const & mt, MPI::Cartcomm const & comm) 
        : win(MPI::Win::Create(mt.data(), static_cast<MPI::Aint>(mt.size()), sizeof(double), MPI::INFO_NULL, comm))
        , comm(comm)
    {}

    void Get(RegularMatrix & to, size_t target_rank) const {
        auto data_size = static_cast<int>(to.size());
        win.Get(to.data(), data_size, MPI_DOUBLE, static_cast<int>(target_rank), 0, data_size, MPI_DOUBLE);
    }

    void Put(RegularMatrix & from, size_t target_rank) const {
        auto data_size = static_cast<int>(from.size());
        win.Put(from.data(), data_size, MPI_DOUBLE, static_cast<int>(target_rank), 0, data_size, MPI_DOUBLE);
    }

    void Fence() const {
        win.Fence(0);
    }
};

struct WinVector : std::vector<Win> {
    void FenceAll() const {
        for (auto const & win : *this)
            win.Fence();
    }
};
