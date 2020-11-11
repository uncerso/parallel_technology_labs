#include <stdio.h>
#include <chrono>
#include <cmath>

double foo(int n) {
    int i;
    double h, sum, x;
    h = 1.0 / (double) n;
    sum = 0.0;
    for (i = 0; i < n; ++i) {
        x = h * ((double) i - 0.5);
        sum += (4.0 / (1.0 + x * x));
    }
    return h * sum;
}

int main() {
    double t = 0,tt = 0;
    size_t tk = 1e5;
    double ans = 0;
    for (size_t k = 0; k < tk; ++k) {
        auto tstart = std::chrono::high_resolution_clock::now();
        double pi = foo(1e5); // run exapmle code
        auto tend = std::chrono::high_resolution_clock::now();
        ans += pi;
        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(tend-tstart).count() / 1e3;
        t += total_time;
        tt += total_time * total_time;
    }
    printf("%.016f\n", ans/tk);
    printf("Mtime: %0.6f\n", t / tk);
    printf("Dtime: %0.6f\n", std::sqrt(tt / tk - (t/tk) * (t/tk)));
}