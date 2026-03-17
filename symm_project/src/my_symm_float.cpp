#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include "../include/symm_implementation.hpp"

using namespace std;
using namespace chrono;

float test_my_symm(int m, int n, int num_threads) {
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
    
    SymmImplementation<float>::symm_parallel(
        Side::LEFT, UpLo::UPPER, m, n,
        alpha, A.data(), m, B.data(), n,
        beta, C.data(), n, num_threads
    );
    
    auto end = high_resolution_clock::now();
    return duration<float>(end - start).count();
}

int main() {
    cout << "Моя реализация SYMM - ОДИНАРНАЯ ТОЧНОСТЬ\n";
    int m = 1500, n = 1500;
    cout << "Размер матриц: " << m << " x " << n << "\n\n";
    
    cout << "------------------------------------------------\n";
    cout << "| Потоки | Попытка |    Время (сек)    |\n";
    cout << "------------------------------------------------\n";
    
    int threads[] = {1, 2, 4, 8, 16};
    
    for (int t : threads) {
        for (int run = 0; run < 10; run++) {
            float time = test_my_symm(m, n, t);
            cout << "|   " << setw(2) << t << "    |   " << setw(2) << run + 1 
                 << "    |   " << fixed << setprecision(4) << setw(10) << time << "    |\n";
        }
        cout << "------------------------------------------------\n\n";
    }
    
    return 0;
}