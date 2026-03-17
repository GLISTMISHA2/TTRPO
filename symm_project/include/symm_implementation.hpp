#ifndef SYMM_IMPLEMENTATION_HPP
#define SYMM_IMPLEMENTATION_HPP

#include <vector>
#include <thread>
#include <algorithm>

enum class Side { LEFT, RIGHT };
enum class UpLo { UPPER, LOWER };

template<typename T>
class SymmImplementation {
public:
    static void symm_parallel(
        Side side, UpLo uplo,
        int m, int n,
        T alpha,
        const T* A, int lda,
        const T* B, int ldb,
        T beta,
        T* C, int ldc,
        int num_threads) {
        
        // Применяем beta к C
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                C[i * ldc + j] *= beta;
            }
        }
        
        if (side == Side::LEFT) {
            symm_left_parallel(uplo, m, n, alpha, A, lda, B, ldb, C, ldc, num_threads);
        } else {
            symm_right_parallel(uplo, m, n, alpha, A, lda, B, ldb, C, ldc, num_threads);
        }
    }

private:
    static void symm_left_parallel(
        UpLo uplo, int m, int n, T alpha,
        const T* A, int lda,
        const T* B, int ldb,
        T* C, int ldc,
        int num_threads) {
        
        std::vector<std::thread> threads;
        int rows_per_thread = m / num_threads;
        
        for (int t = 0; t < num_threads; ++t) {
            int start_row = t * rows_per_thread;
            int end_row = (t == num_threads - 1) ? m : start_row + rows_per_thread;
            
            threads.emplace_back([=, &A, &B, &C]() {
                for (int i = start_row; i < end_row; ++i) {
                    for (int j = 0; j < n; ++j) {
                        T temp = 0;
                        for (int k = 0; k < m; ++k) {
                            T aik;
                            if (uplo == UpLo::UPPER) {
                                aik = (k >= i) ? A[i * lda + k] : A[k * lda + i];
                            } else {
                                aik = (k <= i) ? A[i * lda + k] : A[k * lda + i];
                            }
                            temp += aik * B[k * ldb + j];
                        }
                        C[i * ldc + j] += alpha * temp;
                    }
                }
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
    }
    
    static void symm_right_parallel(
        UpLo uplo, int m, int n, T alpha,
        const T* A, int lda,
        const T* B, int ldb,
        T* C, int ldc,
        int num_threads) {
        
        std::vector<std::thread> threads;
        int rows_per_thread = m / num_threads;
        
        for (int t = 0; t < num_threads; ++t) {
            int start_row = t * rows_per_thread;
            int end_row = (t == num_threads - 1) ? m : start_row + rows_per_thread;
            
            threads.emplace_back([=, &A, &B, &C]() {
                for (int i = start_row; i < end_row; ++i) {
                    for (int j = 0; j < n; ++j) {
                        T temp = 0;
                        for (int k = 0; k < n; ++k) {
                            T ajk;
                            if (uplo == UpLo::UPPER) {
                                ajk = (k >= j) ? A[j * lda + k] : A[k * lda + j];
                            } else {
                                ajk = (k <= j) ? A[j * lda + k] : A[k * lda + j];
                            }
                            temp += B[i * ldb + k] * ajk;
                        }
                        C[i * ldc + j] += alpha * temp;
                    }
                }
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
    }
};

#endif // SYMM_IMPLEMENTATION_HPP
