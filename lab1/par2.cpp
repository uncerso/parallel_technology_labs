#include <stdio.h>
#include <chrono>
#include <cmath>
#include <omp.h>

void integral1(double&);
void integral2(double&);
void integral3(double&);

int main(int argc, char const *argv[]) {
    double t = 0,tt = 0;
    size_t tk = 20;
    double ans = 0;
    omp_set_num_threads(3);
    for (size_t k = 0; k < tk; ++k) {
        auto tstart = std::chrono::high_resolution_clock::now();
        double x = 0, y = 0, z = 0;
        #pragma omp parallel
        #pragma omp sections
        {
            #pragma omp section
            { integral1(x); }
            #pragma omp section
            { integral2(y); }
            #pragma omp section
            { integral3(z); }
        }
        auto tend = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(tend-tstart).count() / 1e3;
        ans += x + y + z;
        t += total_time;
        tt += total_time * total_time;
    }
    printf("%.016f\n", ans/tk);
    printf("Mtime: %0.6f\n", t / tk);
    printf("Dtime: %0.6f\n", std::sqrt(tt / tk - (t/tk) * (t/tk)));
}

void integral1(double& sum) {
    int n = 1e9, i;
    double h, x;
    h = 1.0 / (double) n;
    sum = 0.0;
    for (i = 0; i < n; ++i) {
        x = h * ((double) i - 0.5);
        sum += (4.0 / (1.0 + x * x));
    }
}

void integral2(double& sum) {
    int n = 1e9, i;
    double h, x;
    h = 1.0 / (double) n;
    sum = 0.0;
    for (i = 0; i < n; ++i) {
        x = h * ((double) i - 0.5);
        sum += (4.0 / (1.0 + x * x * x));
    }
}

void integral3(double& sum) {
    int n = 1e9, i;
    double h, x;
    h = 1.0 / (double) n;
    sum = 0.0;
    for (i = 0; i < n; ++i) {
        x = h * ((double) i - 0.5);
        sum += (4.0 / (2.0 + x * x * x));
    }
}
