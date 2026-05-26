#include <cblas.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "symm_project/include/symm_implementation.hpp"

using namespace std;

template<typename T>
bool is_close(T a, T b, T rel_tol = 1e-3, T abs_tol = 1e-3) {
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol);
}

template<typename T>
bool test_symm_correctness(int m, int n, Side side, UpLo uplo, T alpha, T beta) {
    int symm_dim = (side == Side::LEFT) ? m : n;
    vector<T> A(symm_dim * symm_dim);
    vector<T> B(m * n);
    vector<T> C_ref(m * n);
    vector<T> C_my(m * n);

    random_device rd;
    mt19937 gen(42);
    uniform_real_distribution<T> dist(-1.0, 1.0);

    for (int i = 0; i < symm_dim; ++i) {
        for (int j = i; j < symm_dim; ++j) {
            T val = dist(gen);
            A[i * symm_dim + j] = val;
            A[j * symm_dim + i] = val;
        }
    }

    for (int i = 0; i < m * n; ++i) B[i] = dist(gen);
    for (int i = 0; i < m * n; ++i) C_ref[i] = dist(gen);
    for (int i = 0; i < m * n; ++i) C_my[i] = C_ref[i];

    CBLAS_SIDE cblas_side = (side == Side::LEFT) ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == UpLo::UPPER) ? CblasUpper : CblasLower;

    cblas_ssymm(CblasRowMajor, cblas_side, cblas_uplo,
                m, n, alpha, A.data(), symm_dim,
                B.data(), n, beta, C_ref.data(), n);

    SymmImplementation<float>::symm_parallel(
        side, uplo, m, n, alpha, A.data(), symm_dim,
        B.data(), n, beta, C_my.data(), n, 4
    );

    int errors = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!is_close(C_ref[i * n + j], C_my[i * n + j])) {
                errors++;
                if (errors <= 5) {
                    cerr << "Mismatch at [" << i << "][" << j << "]: "
                         << "ref=" << C_ref[i * n + j]
                         << " my=" << C_my[i * n + j] << endl;
                }
            }
        }
    }

    return errors == 0;
}

int main() {
    cout << "=== Correctness test: my_symm vs OpenBLAS ===" << endl;
    cout << "Type: float, tolerance: 1e-3" << endl << endl;

    struct TestCase {
        int m, n;
        Side side;
        UpLo uplo;
        float alpha, beta;
        const char* desc;
    };

    TestCase cases[] = {
        { 4, 4, Side::LEFT,  UpLo::UPPER, 1.0f, 0.0f, "L/U, a=1, b=0 (4x4)" },
        { 4, 4, Side::LEFT,  UpLo::LOWER, 1.0f, 0.0f, "L/L, a=1, b=0 (4x4)" },
        { 4, 4, Side::RIGHT, UpLo::UPPER, 1.0f, 0.0f, "R/U, a=1, b=0 (4x4)" },
        { 4, 4, Side::RIGHT, UpLo::LOWER, 1.0f, 0.0f, "R/L, a=1, b=0 (4x4)" },
        { 5, 3, Side::LEFT,  UpLo::UPPER, 1.0f, 1.0f, "L/U, a=1, b=1 (5x3)" },
        { 3, 5, Side::RIGHT, UpLo::LOWER, 2.0f, 0.5f, "R/L, a=2, b=0.5 (3x5)" },
        { 7, 7, Side::LEFT,  UpLo::UPPER, 0.5f, 1.0f, "L/U, a=0.5, b=1 (7x7)" },
        { 6, 4, Side::RIGHT, UpLo::UPPER, 1.5f, 0.0f, "R/U, a=1.5, b=0 (6x4)" },
    };

    int passed = 0, failed = 0;
    for (auto& tc : cases) {
        bool ok = test_symm_correctness(tc.m, tc.n, tc.side, tc.uplo, tc.alpha, tc.beta);
        cout << (ok ? "[PASS]" : "[FAIL]") << "  " << tc.desc << endl;
        if (ok) passed++; else failed++;
    }

    cout << "\n" << passed << "/" << (passed + failed) << " tests passed." << endl;

    int m = 50, n = 50;
    cout << "\n--- Larger test: m=" << m << " n=" << n << " ---" << endl;
    bool big_ok = test_symm_correctness(50, 50, Side::LEFT, UpLo::UPPER, 1.0f, 0.0f);
    cout << (big_ok ? "[PASS]" : "[FAIL]") << "  L/U, a=1, b=0 (50x50)" << endl;

    int total_ok = (passed + failed > 0 && failed == 0 && big_ok) ? 0 : 1;
    return total_ok;
}
