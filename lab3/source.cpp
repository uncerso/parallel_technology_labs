#include "helpers.hpp"
#include "benchmark.hpp"
#include <iostream>
#include <random>
#include <memory>
#include <exception>
#include <string_view>

using namespace std;
constexpr int dim_size = 2;

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
    comms.grid_size = grid_size;

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

struct BlockMatrixPack {
    unique_ptr<BlockMatrix> a, b, c;
};

BlockMatrixPack GenMatrixPack(bool dummied, size_t block_size, size_t grid_size) {
    BlockMatrixPack pack;
    if (!dummied) {
        size_t matrix_size = block_size * grid_size;
        pack.a = make_unique<BlockMatrix>(GetRandomMatrix(matrix_size), block_size);
        pack.b = make_unique<BlockMatrix>(GetRandomMatrix(matrix_size), block_size);
        pack.c = make_unique<BlockMatrix>(grid_size, block_size);
    } else {
        pack.a = make_unique<BlockMatrix>(grid_size, block_size);
        pack.b = make_unique<BlockMatrix>(grid_size, block_size);
        pack.c = make_unique<BlockMatrix>(grid_size, block_size);
    }

    return pack;
}

void LoadData(State & st, BlockMatrixPack & pack, Comms const & comms, ProcInfo const & proc_info) {
    auto ba_refs = GetRefs(*pack.a, comms.grid);
    auto bb_refs = GetRefs(*pack.b, comms.grid);

    auto lin_address = proc_info.y * comms.grid_size + proc_info.x;
    ba_refs[lin_address].Get(st.a, 0);
    bb_refs[lin_address].Get(*st.b1, 0);
    ba_refs.FenceAll();
    bb_refs.FenceAll();
}

void StoreData(State & st, BlockMatrixPack & pack, Comms const & comms, ProcInfo const & proc_info) {
    auto lin_address = proc_info.y * comms.grid_size + proc_info.x;
    auto bc_refs = GetRefs(*pack.c, comms.grid);
    bc_refs[lin_address].Put(st.c, 0);
    bc_refs.FenceAll();
}

void Multiply(State & st, Comms const & comms, ProcInfo const & proc_info) {
    size_t grid_size = comms.grid_size;
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
}

ostream & operator << (ostream & out, Measured_time const & t) {
    out << "Mean time: " << t.mean_time << " ms, Standard deviation: " << t.standard_deviation << " ms";
    return out;
}

struct Args {
    bool run_mpi;
    size_t matrix_size;
    size_t measure_times;
    Args(int argc, char *argv[]) {
        if (argc != 4)
            throw std::invalid_argument("Usage: mpiexec -n <#processes> run <use_mpi:0|1> <matrix size: uint> <measure times: uint>");
        run_mpi = string_view(argv[1]) == "1";
        matrix_size = stoull(argv[2]);
        measure_times = stoull(argv[3]);
    }
};

void NoMPIRun(Args args) {
    auto a = GetRandomMatrix(args.matrix_size);
    auto b = GetRandomMatrix(args.matrix_size);
    RegularMatrix c(args.matrix_size);
    auto time = measure([&a, &b, c] () mutable {
        c.AddProductOf(a, b);
    }, args.measure_times);
    cout << time << endl;
}

int main(int argc, char *argv[]) {
    InitGuard mpi_guard(argc, argv);
    try {
        Args args(argc, argv);
        ProcInfo proc_info;
        auto proc_num = static_cast<size_t>(MPI::COMM_WORLD.Get_size());
        proc_info.rank = MPI::COMM_WORLD.Get_rank();
        if (!args.run_mpi) {
            if (proc_info.rank == 0)
                NoMPIRun(args);
            return 0;
        }

        auto grid_size = static_cast<size_t>(sqrt(static_cast<double>(proc_num)));
        if (proc_num != grid_size*grid_size) {
            if (proc_info.rank == 0)
                cout << "Number of processes must be a perfect square\n";
            return 0;
        }

        size_t block_size = args.matrix_size / grid_size;

        Comms comms = CreateGridCommunicators(grid_size);
        tie(proc_info.y, proc_info.x) = GetCoords(comms.grid, proc_info.rank);

        State st(block_size);
        auto pack = GenMatrixPack(proc_info.rank, block_size, grid_size);

        comms.grid.Barrier();

        auto time = measure([&st, &pack, &comms, &proc_info] {
            LoadData(st, pack, comms, proc_info);
            comms.grid.Barrier();
            Multiply(st, comms, proc_info);
            StoreData(st, pack, comms, proc_info);
            comms.grid.Barrier();
        }, args.measure_times);

        if (proc_info.rank == 0)
            cout << time << endl;
    } catch (exception const & e) {
        cerr << "Error: "<< e.what() << '\n';
        return 0;
    } catch (...) {
        cerr << "Unknown error :(\n";
        return 0;
    }
    return 0;
}