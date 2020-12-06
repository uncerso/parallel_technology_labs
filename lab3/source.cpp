#include "matrixes.hpp"
#include <iostream>
#include <random>
#include <memory>
#include <mpi.h>
#include <iomanip>
#include <exception>

#include <thread>
#include <chrono>

using namespace std;
constexpr int dim_size = 2;

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
    MPI::Cartcomm grid;
    MPI::Cartcomm col;
    MPI::Cartcomm row;
};

struct State {
    RegularMatrix persistent_a, persistent_b;
    RegularMatrix a, b, c;
    State(size_t n) 
        : persistent_a(n)
        , persistent_b(n)
        , a(n)
        , b(n)
        , c(n)
    {}
};

// generate random matrix with elements from `-max_value` to `max_value`
RegularMatrix GetRandomMatrix(size_t n, size_t max_value = 99) {
    RegularMatrix mt(n);
    random_device rd;
    mt19937 next_rand(rd());
    
    size_t bound = max_value * 2 + 1;

    for (auto & el : mt)
        el = static_cast<double>(next_rand() % bound) - static_cast<double>(max_value);
    return mt;
}

Comms CreateGridCommunicators(size_t grid_size) {
    Comms comms;
    int dims[dim_size] = {static_cast<int>(grid_size), static_cast<int>(grid_size)};
    bool periodic[dim_size] = {false, false};
    
    comms.grid = MPI::COMM_WORLD.Create_cart(dim_size, dims, periodic, true);

    bool row_subdims[2] = {false, true};
    comms.row = comms.grid.Sub(row_subdims);
    
    bool col_subdims[2] = {true, false};
    comms.col = comms.grid.Sub(col_subdims);
    return comms;
}

ostream & operator<<(ostream & out, RegularMatrix const & mt) {
    for (size_t i = 0; i < mt.n; ++i) {
        for (size_t j = 0; j < mt.n; ++j)
            out << setw(4) << mt(i, j) << ' ';
        out << '\n';
    }
    return out;
}

ostream & operator<<(ostream & out, BlockMatrix const & mt) {
    constexpr size_t w = 4;
    auto n = mt.block_size;
    auto line = string(1 + mt[0].size() * (2 + (w + 1) * n),'-');
    cout << line << '\n';
    for (auto const & block_row : mt) {
        for (size_t i = 0; i < n; ++i) {
            cout << "| ";
            for (auto const & block : block_row) {
                for (size_t j = 0; j < n; ++j)
                    cout << setw(w) << block(i, j) << ' ';
                cout << "| ";
            }
            cout << '\n';
        }
        cout << line << '\n';
    }
    return out;
}

void FillRefs(vector<MPI::Win> & refs, BlockMatrix & mt, MPI::Cartcomm const & comm) {
    refs.reserve(mt.size() * mt.size());
    for (auto & row : mt) {
        for (auto & block : row) {
            refs.push_back(MPI::Win::Create(block.data(), static_cast<MPI::Aint>(block.size()), sizeof(double), MPI::INFO_NULL, comm));
            refs.back().Fence(0);
        }
    }
}

vector<MPI::Win> GetRefs(unique_ptr<BlockMatrix> & mt, MPI::Cartcomm const & comm) {
    vector<MPI::Win> refs;
    if (mt)
        FillRefs(refs, *mt, comm);
    else {
        refs.resize(mt->size() * mt->size());
        for (auto & win : refs) {
            char c;
            win = MPI::Win::Create(&c, 1, 1, MPI::INFO_NULL, comm);
            win.Fence(0);
        }
    }
    return refs;
}

pair<size_t, size_t> GetCoords(MPI::Cartcomm const & cc, int rank) {
    int coords[dim_size];
    cc.Get_coords(rank, dim_size, coords);
    return {coords[0], coords[1]};
}

int main(int argc, char *argv[]) {
    InitGuard mpi_guard(argc, argv);
    auto proc_num = static_cast<size_t>(MPI::COMM_WORLD.Get_size());
    ProcInfo proc_info;
    proc_info.rank = MPI::COMM_WORLD.Get_rank();

    auto grid_size = static_cast<size_t>(sqrt(static_cast<double>(proc_num)));
    if (proc_num != grid_size*grid_size) {
        if (proc_info.rank == 0)
            cout << "Number of processes must be a perfect square\n";
        return 0;
    }

    if (proc_info.rank == 0)
        cout << "Parallel matrix multiplication program" << endl;

    Comms comms = CreateGridCommunicators(grid_size);
    tie(proc_info.y, proc_info.x) = GetCoords(comms.grid, proc_info.rank);

    size_t matrix_size = grid_size*2;
    size_t block_size = matrix_size / grid_size;
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
        auto ba_refs = GetRefs(ba, comms.grid);
        auto bb_refs = GetRefs(bb, comms.grid);
        auto bc_refs = GetRefs(bc, comms.grid);

        State st(block_size);

        auto lin_address = proc_info.y * grid_size + proc_info.x;
        ba_refs[lin_address].Get(st.a.data(), static_cast<int>(st.a.size()), MPI_DOUBLE, 0, 0, static_cast<int>(st.a.size()), MPI_DOUBLE);
        bb_refs[lin_address].Get(st.b.data(), static_cast<int>(st.b.size()), MPI_DOUBLE, 0, 0, static_cast<int>(st.b.size()), MPI_DOUBLE);
        for (auto & win : ba_refs)
            win.Fence(0);
        for (auto & win : bb_refs)
            win.Fence(0);

        st.persistent_a = st.a;
        st.persistent_b = st.b;
        auto pers_a_win = MPI::Win::Create(st.persistent_a.data(), static_cast<MPI::Aint>(st.persistent_a.size()), sizeof(double), MPI::INFO_NULL, comms.row);
        auto pers_b_win = MPI::Win::Create(st.persistent_b.data(), static_cast<MPI::Aint>(st.persistent_b.size()), sizeof(double), MPI::INFO_NULL, comms.col);
        pers_a_win.Fence(0);
        pers_b_win.Fence(0);
        for (size_t i = 0; i < grid_size; ++i) {
            int pos = static_cast<int>((proc_info.y + i)%grid_size);
            pers_a_win.Get(st.a.data(), static_cast<int>(st.a.size()), MPI_DOUBLE, pos, 0, static_cast<int>(st.a.size()), MPI_DOUBLE);
            pers_a_win.Fence(0);
            st.c.AddProductOf(st.a, st.b);
            pos = static_cast<int>((proc_info.y + 1)%grid_size);
            pers_b_win.Get(st.b.data(), static_cast<int>(st.b.size()), MPI_DOUBLE, pos, 0, static_cast<int>(st.a.size()), MPI_DOUBLE);
            pers_b_win.Fence(0);
            st.persistent_b = st.b;
        }

        bc_refs[lin_address].Put(st.c.data(), static_cast<int>(st.c.size()), MPI_DOUBLE, 0, 0, static_cast<int>(st.c.size()), MPI_DOUBLE);
        for (auto & win : bc_refs)
            win.Fence(0);
        if (proc_info.rank == 0) {
            BlockMatrix bm(*c, block_size);
            cout << (bm == *bc) << endl;
        }
    } catch (exception const & e) {
        cerr << e.what() << '\n';
        return 0;
    } catch (...) {
        cerr << "Unknown error :(\n";
        return 0;
    }
    return 0;
}