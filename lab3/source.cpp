#include "matrixes.hpp"
#include <iostream>
#include <random>
#include <memory>
#include <mpi.h>
#include <exception>

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
    RegularMatrix persistent_a;
    RegularMatrix a, c;
    unique_ptr<RegularMatrix> b1, b2;
    State(size_t n) 
        : persistent_a(n)
        , a(n)
        , c(n)
        , b1(make_unique<RegularMatrix>(n))
        , b2(make_unique<RegularMatrix>(n))
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

struct WinVector : vector<Win> {
    void FenceAll() const {
        for (auto const & win : *this)
            win.Fence();
    }
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

WinVector GetRefs(BlockMatrix const & mt, MPI::Cartcomm const & comm) {
    WinVector refs;
    refs.reserve(mt.size() * mt.size());
    for (auto & row : mt) {
        for (auto & block : row) {
            refs.emplace_back(block, comm);
            refs.back().Fence();
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

    size_t matrix_size = grid_size*15;
    size_t block_size = matrix_size / grid_size;
    try {
        unique_ptr<RegularMatrix> c;
        unique_ptr<BlockMatrix> ba, bb, bc;
        if (proc_info.rank == 0) {
            auto a = make_unique<RegularMatrix>(GetRandomMatrix(matrix_size));
            auto b = make_unique<RegularMatrix>(GetRandomMatrix(matrix_size));
            c = make_unique<RegularMatrix>(matrix_size);
            ba = make_unique<BlockMatrix>(*a, block_size);
            bb = make_unique<BlockMatrix>(*b, block_size);
            bc = make_unique<BlockMatrix>(grid_size, block_size);
            c->AddProductOf(*a, *b);
        } else {
            ba = make_unique<BlockMatrix>(grid_size, block_size);
            bb = make_unique<BlockMatrix>(grid_size, block_size);
            bc = make_unique<BlockMatrix>(grid_size, block_size);
        }
        auto ba_refs = GetRefs(*ba, comms.grid);
        auto bb_refs = GetRefs(*bb, comms.grid);
        auto bc_refs = GetRefs(*bc, comms.grid);

        State st(block_size);

        auto lin_address = proc_info.y * grid_size + proc_info.x;
        ba_refs[lin_address].Get(st.a, 0);
        bb_refs[lin_address].Get(*st.b1, 0);
        ba_refs.FenceAll();
        bb_refs.FenceAll();

        st.persistent_a = st.a;
        Win a_win(st.persistent_a, comms.row);
        auto b1_win = make_unique<Win>(*st.b1, comms.col);
        auto b2_win = make_unique<Win>(*st.b2, comms.col);
        a_win.Fence();
        b1_win->Fence();
        b2_win->Fence();
        for (size_t i = 0; i < grid_size; ++i) {
            a_win.Get(st.a, (proc_info.y + i) % grid_size);
            a_win.Fence();

            st.c.AddProductOf(st.a, *st.b1);

            b1_win->Get(*st.b2, (proc_info.y + 1) % grid_size);
            b1_win->Fence();
            swap(st.b1, st.b2);
            swap(b1_win, b2_win);
            comms.col.Barrier();
        }

        bc_refs[lin_address].Put(st.c, 0);
        bc_refs.FenceAll();

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