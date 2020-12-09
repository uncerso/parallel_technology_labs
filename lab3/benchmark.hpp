#include <functional>
#include <chrono>
#include <cmath>

struct Measured_time {
    double mean_time;
    double standard_deviation;
};


Measured_time measure(std::function<void()> const & func, size_t amount_of_runs) {
    using namespace std;
    Measured_time tm = {0.0, 0.0};
    for (size_t i = 0; i < amount_of_runs; ++i) {
        auto test_func = func;
        auto time_point1 = chrono::high_resolution_clock::now();
        test_func();
        auto time_point2 = chrono::high_resolution_clock::now();
        auto t = static_cast<double>(chrono::duration_cast<chrono::microseconds>(time_point2 - time_point1).count()) / 1e3;
        tm.mean_time += t;
        tm.standard_deviation += t * t;
    }
    auto tk = static_cast<double>(amount_of_runs);
    tm.mean_time /= tk;
    tm.standard_deviation = std::sqrt(tm.standard_deviation / tk - tm.mean_time * tm.mean_time);
    return tm;
}
