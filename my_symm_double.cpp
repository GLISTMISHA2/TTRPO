#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include "../include/symm_implementation.hpp"

using namespace std;
using namespace chrono;

double test_my_symm(int m, int n, int num_threads) {
    vector<double> A(m * m);
    vector<double> B(m * n);
    vector<double> C(m * n, 0.0);
    
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) {
            double val = dist(gen);
            A[i * m + j] = val;
            A[j * m + i] = val;
        }
    }
    
    for (int i = 0; i < m * n; ++i) B[i] = dist(gen);
    for (int i = 0; i < m * n; ++i) C[i] = dist(gen);
    
    double alpha = 1.0, beta = 1.0;
    
    auto start = high_resolution_clock::now();
    
    SymmImplementation<double>::symm_parallel(
        Side::LEFT, UpLo::UPPER, m, n,
        alpha, A.data(), m, B.data(), n,
        beta, C.data(), n, num_threads
    );
    
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

int main() {
    cout << "Моя реализация SYMM - ДВОЙНАЯ ТОЧНОСТЬ\n";
    int m = 1500, n = 1500;
    cout << "Размер матриц: " << m << " x " << n << "\n\n";
    
    cout << "------------------------------------------------\n";
    cout << "| Потоки | Попытка |    Время (сек)    |\n";
    cout << "------------------------------------------------\n";
    
    int threads[] = {1, 2, 4, 8, 16};
    
    for (int t : threads) {
        for (int run = 0; run < 10; run++) {
            double time = test_my_symm(m, n, t);
            cout << "|   " << setw(2) << t << "    |   " << setw(2) << run + 1 
                 << "    |   " << fixed << setprecision(4) << setw(10) << time << "    |\n";
        }
        cout << "------------------------------------------------\n\n";
    }
    
    return 0;
}
