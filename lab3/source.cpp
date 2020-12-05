#include "matrixes.hpp"
#include <iostream>
#include <random>
#include <memory>
#include <mpi.h>
#include <iomanip>
#include <exception>

using namespace std;
constexpr int dim_size = 2;

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
    union {
        struct {
            int x = 0;
            int y = 0;
        };
        int coords[2];
    };
};

struct Comms {
    MPI_Comm grid;
    MPI_Comm col;
    MPI_Comm row;
};

struct State {
    RegularMatrix persistent_a;
    RegularMatrix a, b, c;
    State(size_t n) 
        : persistent_a(n)
        , a(n)
        , b(n)
        , c(n)
    {}
};

// generate random matrix with elements from `-max_value` to `max_value`
RegularMatrix GetRandomMatrix(size_t n, size_t max_value = 9) {
    RegularMatrix mt(n);
    random_device rd;
    mt19937 next_rand(rd());
    
    size_t bound = max_value * 2 + 1;

    for (auto & el : mt)
        el = static_cast<double>(next_rand() % bound) - static_cast<double>(max_value);
    return mt;
}

Comms CreateGridCommunicators(int grid_size) {
    Comms comms;
    int dims[dim_size] = {grid_size, grid_size};
    int periodic[dim_size] = {0, 0};
    
    MPI_Cart_create(MPI_COMM_WORLD, dim_size, dims, periodic, 1, &comms.grid);

    int row_subdims[2] = {0, 1};
    MPI_Cart_sub(comms.grid, row_subdims, &comms.row);
    
    int col_subdims[2] = {1, 0};
    MPI_Cart_sub(comms.grid, col_subdims, &comms.col);
    return comms;
}

ostream & operator<<(ostream & out, RegularMatrix const & mt) {
    for (size_t i = 0; i < mt.n; ++i) {
        for (size_t j = 0; j < mt.n; ++j)
            out << setw(3) << mt(i, j) << ' ';
        out << '\n';
    }
    return out;
}

ostream & operator<<(ostream & out, BlockMatrix const & mt) {
    for (auto const & block_row : mt) {
        for (auto const & block : block_row) {
            cout << block << '\n';
        }
    }
    return out;
}

void FillRefs(vector<MPI_Win> & refs, BlockMatrix & mt, MPI_Comm comm) {
    refs.reserve(mt.size() * mt.size());
    for (auto & row : mt) {
        for (auto & block : row) {
            MPI_Win win;
            cerr << "!!!\n";
            MPI_Win_create(block.data(), static_cast<int>(block.size()), sizeof(double), MPI_INFO_NULL, comm, &win);
            MPI_Win_fence(0,win);
            cerr << "!!!\n";
            refs.push_back(win);
        }
    }
}

vector<MPI_Win> GetRefs(unique_ptr<BlockMatrix> & mt, MPI_Comm comm) {
    vector<MPI_Win> refs;
    if (mt)
        FillRefs(refs, *mt, comm);
    else {
        refs.resize(mt->size() * mt->size());
        for (auto & win : refs)
            MPI_Win_create(nullptr, 0, 0, MPI_INFO_NULL, comm, &win);
    }
    return refs;
}

int main(int argc, char *argv[]) {
    InitGuard mpi_guard(&argc, &argv);
    int proc_num;
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
    ProcInfo proc_info;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_info.rank);

    auto grid_size = static_cast<int>(sqrt(static_cast<double>(proc_num)));
    if (proc_num != grid_size*grid_size) {
        if (proc_info.rank == 0)
            cout << "Number of processes must be a perfect square\n";
        return 0;
    }

    if (proc_info.rank == 0)
        cout << "Parallel matrix multiplication program\n";

    Comms comms = CreateGridCommunicators(grid_size);
    MPI_Cart_coords(comms.grid, proc_info.rank, dim_size, proc_info.coords);

    cout << "Coords: x: " << proc_info.x << ", y: " << proc_info.y << '\n';
    if (proc_info.rank > 1) return 0;

    size_t matrix_size = static_cast<size_t>(grid_size)*4;
    size_t block_size = matrix_size / static_cast<size_t>(grid_size);
    try {
        unique_ptr<RegularMatrix> a, b, c;
        unique_ptr<BlockMatrix> ba, bb, bc;
        // if (proc_info.rank == 0) {
            a = make_unique<RegularMatrix>(GetRandomMatrix(matrix_size));
            b = make_unique<RegularMatrix>(GetRandomMatrix(matrix_size));
            c = make_unique<RegularMatrix>(matrix_size);
            ba = make_unique<BlockMatrix>(*a, block_size);
            bb = make_unique<BlockMatrix>(*b, block_size);
            bc = make_unique<BlockMatrix>(ba->size(), block_size);
            c->AddProductOf(*a, *b);
        // }
        cerr << "Rank: " << proc_info.rank << " 0\n";
        MPI_Win win;
        double d;
        MPI_Win_create(&d, 1, sizeof(d), MPI_INFO_NULL, comms.grid, &win);
        cerr << "Rank: " << proc_info.rank << " 1\n";
        MPI_Win_fence(0,win);
        cerr << "Rank: " << proc_info.rank << " 2\n";
        return 0;
        auto ba_refs = GetRefs(ba, comms.grid);
        auto bb_refs = GetRefs(ba, comms.grid);
        auto bc_refs = GetRefs(bc, comms.grid);

        cerr << "Rank: " << proc_info.rank << " 2\n";
        State st(block_size);

        st.c[0] = proc_info.rank;

        auto lin_address = static_cast<size_t>(proc_info.y * grid_size + proc_info.x);
        MPI_Win_fence(0,ba_refs[lin_address]);
        MPI_Win_fence(0,bb_refs[lin_address]);
        MPI_Win_fence(0,bc_refs[lin_address]);
        cerr << "Rank: " << proc_info.rank << " 3\n";
        MPI_Get(st.a.data(), static_cast<int>(st.a.size()), MPI_DOUBLE, 0, 0, static_cast<int>(st.a.size()), MPI_DOUBLE, ba_refs[lin_address]);
        MPI_Get(st.b.data(), static_cast<int>(st.b.size()), MPI_DOUBLE, 0, 0, static_cast<int>(st.b.size()), MPI_DOUBLE, bb_refs[lin_address]);
        MPI_Put(st.c.data(), static_cast<int>(st.c.size()), MPI_DOUBLE, 0, 0, static_cast<int>(st.c.size()), MPI_DOUBLE, bc_refs[lin_address]);
        cerr << "Rank: " << proc_info.rank << " 4\n";
        MPI_Win_fence(0,ba_refs[lin_address]);
        MPI_Win_fence(0,bb_refs[lin_address]);
        MPI_Win_fence(0,bc_refs[lin_address]);
        cerr << "Rank: " << proc_info.rank << " 5\n";
        st.persistent_a = st.a;
        if (proc_info.rank == 0)
            cout << (BlockMatrix(*c, block_size) == *bc) << '\n';
    } catch (exception const & e) {
        cerr << e.what() << '\n';
        return 0;
    } catch (...) {
        cerr << "Unknown error :(\n";
        return 0;
    }
    return 0;
}