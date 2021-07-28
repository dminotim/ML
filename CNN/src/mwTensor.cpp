#include "mwTensor.hpp"

//void gemm_nn(int M, int N, int K,
//	float* A, int lda,
//	float* B, int ldb,
//	float* C, int ldc)
//{
//	int i, j, k;
//	for (i = 0; i < M; ++i) {
//		float* c = C + i * ldc;
//		for (k = 0; k < K; ++k) {
//			const float* b = B + k * ldb;
//			float a = A[i * lda + k];
//			for (j = 0; j < N; ++j) {
//				c[j] += a * b[j];
//			}
//		}
//	}
//}
//
//void gemm_nn(int M, int N, int K,
//	double* A, int lda,
//	double* B, int ldb,
//	double* C, int ldc)
//{
//	int i, j, k;
//	for (i = 0; i < M; ++i) {
//		for (k = 0; k < K; ++k) {
//			register double A_PART = A[i * lda + k];
//			for (j = 0; j < N; ++j) {
//				C[i * ldc + j] += A_PART * B[k * ldb + j];
//			}
//		}
//	}
//}