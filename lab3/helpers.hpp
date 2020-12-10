#pragma once
#include "matrixes.hpp"
#include <vector>
#include <memory>
#include <mpi.h>

struct InitGuard {
    InitGuard(int *argc, char ***argv) {
        MPI_Init(argc, argv);
    }

    ~InitGuard() {
        MPI_Finalize();
    }
};

struct ProcInfo {
    int rank = 0;
    size_t x = 0;
    size_t y = 0;
};

struct Comms {
    size_t grid_size;
    MPI_Comm grid;
    MPI_Comm col;
    MPI_Comm row;
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
    MPI_Win win;
    MPI_Comm const & comm;
public:
    Win(RegularMatrix const & mt, MPI_Comm const & comm) 
        : comm(comm)
    {
        MPI_Win_create(const_cast<double *>(mt.data()), static_cast<int>(mt.size()), sizeof(double), MPI_INFO_NULL, comm, &win);
    }

    ~Win() {
        MPI_Win_free(&win);
    }

    void Get(RegularMatrix & to, size_t target_rank) const {
        auto data_size = static_cast<int>(to.size());
        MPI_Get(to.data(), data_size, MPI_DOUBLE, static_cast<int>(target_rank), 0, data_size, MPI_DOUBLE, win);
    }

    void Put(RegularMatrix & from, size_t target_rank) const {
        auto data_size = static_cast<int>(from.size());
        MPI_Put(from.data(), data_size, MPI_DOUBLE, static_cast<int>(target_rank), 0, data_size, MPI_DOUBLE, win);
    }

    void Fence() const {
        MPI_Win_fence(0, win);
    }
};

struct WinVector : std::vector<Win> {
    void FenceAll() const {
        for (auto const & win : *this)
            win.Fence();
    }
};
