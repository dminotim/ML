#pragma once
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include <memory>
#include "dmMatrix.hpp"
#include "dmLUDecompose.hpp"

namespace dmLDLDecompose
{

template <class T>
struct dmLDL
{
	std::shared_ptr<dmMatrix<T>> m_L;
	std::shared_ptr<dmMatrix<T>> m_LT;
	std::vector<T> m_d;
};

template <class T>
dmLDL<T> GetLDL(const dmMatrix<T>& matr)
{
	if (matr.GetColCount() != matr.GetRowCount())
		throw std::exception("Input matrix must be square!");
	dmLDL<T> ldlRes;
	ldlRes.m_L =
		std::make_shared<dmMatrix<T>>(matr.GetRowCount(), matr.GetColCount());
	dmMatrix<T>& L = *ldlRes.m_L;
	std::vector<T>& d = ldlRes.m_d;
	d.resize(matr.GetRowCount(), T(0));
	for (size_t i = 0; i < L.GetRowCount(); ++i)
	{
		L(i, i) = T(1);
		T a = matr(i, i);
		for (size_t j = 0; j < i; ++j)
		{
			a -= d[j] * L(i, j) * L(i, j);
		}
		d[i] = a;
		if (d[i] == 0)
			return dmLDL<T>();
		for (size_t j = i + 1; j < d.size(); ++j)
		{
			L(i, j) = 0;
			a = matr(j, i);
			for (size_t k = 0; k < i; ++k)
			{
				a -= d[k] * L(j, k) * L(i, k);
			}
			L(j, i) = a / d[i];
		}
	}
	ldlRes.m_LT = std::make_shared<dmMatrix<T>>(L);
	ldlRes.m_LT->Transpose();
	return ldlRes;
}

template<typename T>
std::vector<T> Solve(const dmMatrix<T>& matr, const std::vector<T>& b)
{
	auto ldl = GetLDL(matr);
	const dmMatrix<T>& L = *ldl.m_L;
	const dmMatrix<T>& LT = *ldl.m_LT;
	const std::vector<T>& d = ldl.m_d;
	const std::vector<T> y = dmLUDecompose::ForwardSubstitution(L, b);
	std::vector<T> z = y;
	for (size_t i = 0; i < z.size(); ++i)
	{
		z[i] /= d[i];
	}
	return dmLUDecompose::BackSubstitution(LT, z);
}

template<typename T>
std::vector<T> Solve(const dmLDL<T>& ldl, const std::vector<T>& b)
{
	const dmMatrix<T>& L = *ldl.m_L;
	const dmMatrix<T>& LT = *ldl.m_LT;
	const std::vector<T>& d = ldl.m_d;
	const std::vector<T> y = dmLUDecompose::ForwardSubstitution(L, b);
	const std::vector<T> z = y;
	for (size_t i = 0; i < z.size(); ++i)
	{
		z[i] /= d[i];
	}
	return dmLUDecompose::BackSubstitution(LT, z);
}

}