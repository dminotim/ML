#pragma once
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include <memory>
#include "dmMatrix.hpp"

namespace dmLUDecompose
{
template <class T>
struct dmLU
{
	std::shared_ptr<dmMatrix<T>> m_L;
	std::shared_ptr<dmMatrix<T>> m_U;
	std::shared_ptr<dmMatrix<int>> m_permutation;
};

template <class T>
dmLU<T> GetLU(const dmMatrix<T>& matr)
{
	if (matr.GetColCount() != matr.GetRowCount())
		throw std::exception("Input matrix must be square!");
	dmLU<T> luRes;
	luRes.m_U = std::make_shared<dmMatrix<T>>(matr);
	luRes.m_L = std::make_shared<dmMatrix<T>>(matr.GetRowCount(), matr.GetColCount());
	luRes.m_permutation = std::make_shared<dmMatrix<int>>(matr.GetColCount(), matr.GetRowCount());

	for (size_t i = 0; i < luRes.m_permutation->GetRowCount(); ++i)
	{
		(*luRes.m_permutation)(i, i) = 1;
	}
	
	T max_val;
	size_t max_index;
	dmMatrix<T>& U = *luRes.m_U;
	dmMatrix<T>& L = *luRes.m_L;
	dmMatrix<int>& P = *luRes.m_permutation;
	for (size_t k = 0; k + 1 < U.GetRowCount(); ++k) {
		max_val = -1;
		max_index = k;

		for (size_t z = k; z < U.GetRowCount(); ++z) {
			const T candidate = std::abs(U(z, k));
			if (candidate > max_val) {
				max_val = candidate;
				max_index = z;
			}
		}

		U.SwapRows(k, max_index);
		P.SwapRows(k, max_index);
		L.SwapRows(k, max_index);

		T sum;
		for (size_t i = k + 1; i < U.GetRowCount(); ++i) {
			sum = U(i, k) / U(k, k);
			for (size_t j = k; j < U.GetColCount(); ++j) {
				U(i, j) -= sum * U(k, j);
			}
			L(i, k) = sum;
		}
	}
	for (size_t i = 0; i < U.GetRowCount(); ++i) {
		L(i, i) += 1;
	}
	P.Transpose();
	*luRes.m_U = luRes.m_U->UpperTriangular();
	return luRes;
}

template <class T>
std::vector<T> BackSubstitution(const dmMatrix<T>& matr, const std::vector<T>& b)
{
	if (matr.m_colCount != matr.m_rowCount)
		throw std::exception("Input matrix must be square!");
	if (matr.m_colCount != b.size())
		throw std::exception("The dimensions of A and b don't match!");

	std::vector<T> solution(b.size());
	T sum;
	for (int k = b.size() - 1; k >= 0; k--)
	{
		sum = 0;
		for (size_t j = k + 1; j < b.size(); j++)
		{
			sum += matr(k, j) * solution[j];
		}
		solution[k] = (b[k] - sum) / matr(k, k);
	}
	return solution;
}

template <class T>
std::vector<T> ForwardSubstitution(const dmMatrix<T>& matr, const std::vector<T>& b) {
	if (matr.m_colCount != matr.m_rowCount)
		throw std::exception("Input matrix must be square!");
	if (matr.m_colCount != b.size())
		throw std::exception("The dimensions of A and b don't match!");

	std::vector<T> solution(b.size());
	T sum;
	for (size_t k = 0; k < b.size(); k++)
	{
		sum = 0;

		for (size_t j = 0; j < k; j++)
		{
			sum = sum + matr(k, j) * solution[j];
		}
		solution[k] = (b[k] - sum) / matr(k,k);
	}

	return solution;
}

template<typename T>
std::vector<T> Solve(const dmMatrix<T>& matr, const std::vector<T>& b)
{
	auto plu = GetLU(matr);
	const dmMatrix<T>& U = *plu.m_U;
	const dmMatrix<T>& L = *plu.m_L;
	dmMatrix<int> P = *plu.m_permutation;
	P.Transpose();
	const std::vector<T> permutedB = P.Multiply(b);
	const std::vector<T> y = ForwardSubstitution(L, permutedB);
	return BackSubstitution(U, y);
}

template<typename T>
std::vector<T> Solve(const dmLU<T>& lu, const std::vector<T>& b)
{
	const dmMatrix<T>& U = *lu.m_U;
	const dmMatrix<T>& L = *lu.m_L;
	dmMatrix<int> P = *lu.m_permutation;
	P.Transpose();
	const std::vector<T> permutedB = P.Multiply(b);
	const std::vector<T> y = ForwardSubstitution(L, permutedB);
	return BackSubstitution(U, y);
}

}