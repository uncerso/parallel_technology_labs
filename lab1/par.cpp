#include <stdio.h>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <omp.h>

double foo(int n) {
    int i;
    double h, sum, x;
    h = 1.0 / (double) n;
    sum = 0.0;

    #pragma omp parallel
    {
        int numt = omp_get_num_threads();
        double sum2 = 0;
        #pragma omp for
        for (i = 0; i < n; ++i) {
            x = h * ((double) i - 0.5);
            sum += (4.0 / (1.0 + x * x));
        }
        #pragma omp atomic
        sum += sum2;
    }
    return h * sum;
}

int main(int argc, char const *argv[]) {
    if (argc != 2)
        return 1;

    int threads = std::atoi(argv[1]);
    omp_set_num_threads(threads);

    double t = 0,tt = 0;
    size_t tk = 20;
    double ans = 0;
    for (size_t k = 0; k < tk; ++k) {
        auto tstart = std::chrono::high_resolution_clock::now();
        double pi = foo(1e9); // run exapmle code
        auto tend = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(tend-tstart).count() / 1e3;
        ans += pi;
        t += total_time;
        tt += total_time * total_time;
    }
    printf("%.016f\n", ans/tk);
    printf("Mtime: %0.6f\n", t / tk);
    printf("Dtime: %0.6f\n", std::sqrt(tt / tk - (t/tk) * (t/tk)));
}