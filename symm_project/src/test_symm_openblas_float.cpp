#include <cblas.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>

using namespace std;
using namespace chrono;

float test_symm(int m, int n, int num_threads) {
    openblas_set_num_threads(num_threads);
    
    vector<float> A(m * m);
    vector<float> B(m * n);
    vector<float> C(m * n, 0.0f);
    
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) {
            float val = dist(gen);
            A[i * m + j] = val;
            A[j * m + i] = val;
        }
    }
    
    for (int i = 0; i < m * n; ++i) B[i] = dist(gen);
    for (int i = 0; i < m * n; ++i) C[i] = dist(gen);
    
    float alpha = 1.0f, beta = 1.0f;
    
    auto start = high_resolution_clock::now();
    
    cblas_ssymm(CblasRowMajor, CblasLeft, CblasUpper,
                m, n, alpha, A.data(), m,
                B.data(), n, beta, C.data(), n);
    
    auto end = high_resolution_clock::now();
    return duration<float>(end - start).count();
}

int main() {
    cout << "Тестирование производительности SYMM (OpenBLAS) - ОДИНАРНАЯ ТОЧНОСТЬ\n";
    int m = 2000, n = 2000;
    cout << "Размер матриц: " << m << " x " << n << "\n\n";
    
    cout << "------------------------------------------------\n";
    cout << "| Потоки | Попытка |    Время (сек)    |\n";
    cout << "------------------------------------------------\n";
    
    int threads[] = {1, 2, 4, 8, 16};
    
    for (int t : threads) {
        for (int run = 0; run < 10; run++) {
            float time = test_symm(m, n, t);
            cout << "|   " << setw(2) << t << "    |   " << setw(2) << run + 1 
                 << "    |   " << fixed << setprecision(4) << setw(10) << time << "    |\n";
        }
        cout << "------------------------------------------------\n\n";
    }
    
    return 0;
}
