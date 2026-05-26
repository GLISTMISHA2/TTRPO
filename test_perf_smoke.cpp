#include <iostream>
#include <vector>
#include <chrono>
#include <random>

extern "C" {
    #include <cblas.h>
}

using namespace std;
using namespace chrono;

float test_openblas_symm(int m, int n, int threads) {
    openblas_set_num_threads(threads);
    vector<float> A(m * m);
    vector<float> B(m * n);
    vector<float> C(m * n, 0.0f);
    mt19937 gen(42);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : A) v = dist(gen);
    for (auto& v : B) v = dist(gen);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < i; ++j)
            A[j * m + i] = A[i * m + j];
    auto start = high_resolution_clock::now();
    cblas_ssymm(CblasRowMajor, CblasLeft, CblasUpper,
                m, n, 1.0f, A.data(), m, B.data(), n, 0.0f, C.data(), n);
    return duration<float>(high_resolution_clock::now() - start).count();
}

int main() {
    cout << "=== Performance smoke test (CI) ===" << endl;
    for (int t : {1, 2, 4}) {
        float t_sec = test_openblas_symm(200, 200, t);
        cout << "  OpenBLAS symm 200x200, threads=" << t << ": " << t_sec << " sec" << endl;
    }
    cout << "Performance smoke test PASSED" << endl;
    return 0;
}
