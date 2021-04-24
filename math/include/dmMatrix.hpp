#pragma once
#include <vector>
#include <algorithm>
//!  dmMatrix  class is a dense matrix stored row by row
/*!
  
*/


template<class T>
struct dmBlock
{
	dmBlock(T* values, const size_t colCount, const size_t rowCount);
	dmBlock();
	dmBlock<T> Multiply(const dmBlock<T>& factor, dmBlock<T>& res);
	size_t m_colCount;
	size_t m_rowCount;
	T* m_values;
};

template<class T>
dmBlock<T>::dmBlock()
	:m_values(nullptr), m_colCount(0), m_rowCount(0)
{

}

template<class T>
dmBlock<T> dmBlock<T>::Multiply(const dmBlock<T>& factors, dmBlock<T>& res)
{
	if (this->m_colCount != factors.m_rowCount)
		throw std::exception("Invalid matrix size");

	const dmBlock<T>& A = *this;
	const dmBlock<T>& B = factors;
	for (size_t i = 0; i < res.m_rowCount; ++i)
	{
		for (size_t j = 0; j < res.m_colCount; ++j)
		{
			for (size_t k = 0; k < B.m_rowCount; ++k)
			{
				res.m_values[i * res.m_colCount + j] +=
					A.m_values[i * A.m_colCount + k] * B.m_values[k * B.m_colCount + j];
			}
		}
	}
	return res;
}

template<class T>
dmBlock<T>::dmBlock(T* values, const size_t colCount, const size_t rowCount)
	:m_values(values), m_colCount(colCount), m_rowCount(rowCount)
{
}

template<class T>
class dmMatrix
{
public:
	dmMatrix(const size_t rowCount, const size_t colCount);
	~dmMatrix();
	const T& operator()(const size_t i, const size_t j) const;
	T& operator()(const size_t i, const size_t j);
	std::vector<T> Multiply(const std::vector<T>& factors);
	dmMatrix<T> MultiplySlow(const dmMatrix<T>& factors);
	dmMatrix<T> Multiply(const dmMatrix<T>& factors);
	size_t GetColCount() const;
	size_t GetRowCount() const;
	//dmMatrix<T> operator * (const dmMatrix<T>& factor) const;
	dmMatrix<dmBlock<T>> ToBlocked(const dmMatrix<T>& matr, std::vector<T>& memory, const size_t blockSize);
	std::vector<T> m_values;
	size_t m_rowCount;
	size_t m_colCount;
private:


};

template<class T>
dmMatrix<dmBlock<T>> dmMatrix<T>::ToBlocked(const dmMatrix<T>& matr,
	std::vector<T>& memory, const size_t blockSize)
{
	memory.resize(matr.m_values.size());
	const size_t N = matr.m_rowCount % blockSize == 0
		? matr.m_rowCount / blockSize : matr.m_rowCount / blockSize + 1;
	const size_t M = matr.m_colCount % blockSize == 0
		? matr.m_colCount / blockSize : matr.m_colCount / blockSize + 1;
	dmMatrix<dmBlock<T>> res(N, M);
	size_t id = 0;
	for (size_t i = 0; i < N; ++i)
	{
		const size_t realIStart = i * blockSize;
		const size_t realIEnd = std::min(realIStart + blockSize, matr.m_rowCount);
		for (size_t j = 0; j < M; ++j)
		{
			const size_t realJStart = j * blockSize;
			const size_t realJEnd = std::min(realJStart + blockSize, matr.m_colCount);
			size_t nextId = id;
			for (size_t il = realIStart; il < realIEnd; ++il)
			{
				for (size_t jl = realJStart; jl < realJEnd; ++jl)
				{
					memory[nextId++] = matr.m_values[il * matr.m_colCount + jl];
				}
			}
			res.m_values[i * M + j] = dmBlock<T>(&memory[id], realIEnd - realIStart, realJEnd - realJStart);
			id = nextId;
		}
	}
	return res;
}


template<class T>
dmMatrix<T> dmMatrix<T>::Multiply(const dmMatrix<T>& factors)
{
	if (this->m_colCount != factors.m_rowCount)
		throw std::exception("Invalid matrix size");
	dmMatrix<T> res(this->m_rowCount, factors.m_colCount);
	const dmMatrix<T>& A = *this;
	const dmMatrix<T>& B = factors;

	const int blockSize = 64 / sizeof(T);
	std::vector<T> memoryRes(res.m_values.size());
	std::vector<T> memoryA(A.m_values.size());
	std::vector<T> memoryB(B.m_values.size());
	auto resBlocked = ToBlocked(res, memoryRes, blockSize);
	auto aBlocked = ToBlocked(A, memoryA, blockSize);
	auto bBlocked = ToBlocked(B, memoryB, blockSize);
	std::vector<T> tempPool(blockSize * blockSize);
	for (size_t i = 0; i < resBlocked.m_rowCount; ++i)
	{
		for (size_t j = 0; j < resBlocked.m_colCount; ++j)
		{
			for (size_t k = 0; k < bBlocked.m_rowCount; ++k)
			{
				aBlocked.m_values[i * aBlocked.m_colCount + k].Multiply(
					bBlocked.m_values[k * bBlocked.m_colCount + j],
					resBlocked.m_values[i * resBlocked.m_colCount + j]);
			}
		}
	}
	std::sort(memoryRes.begin(), memoryRes.end());
	std::cout << std::endl;
	for (auto& el : memoryRes)
	{
		std::cout << el << " ";
	}
	std::cout << std::endl;
	return res;
}

template<class T>
size_t dmMatrix<T>::GetRowCount() const
{
	return m_rowCount;
}

template<class T>
size_t dmMatrix<T>::GetColCount() const
{
	return m_colCount;
}

template<class T>
dmMatrix<T> dmMatrix<T>::MultiplySlow(const dmMatrix<T>& factors)
{
	if (this->m_colCount != factors.m_rowCount)
		throw std::exception("Invalid matrix size");
	dmMatrix<T> res(this->m_rowCount, factors.m_colCount);
	const dmMatrix<T>& A = *this;
	const dmMatrix<T>& B = factors;
	for (size_t i = 0; i < res.m_rowCount; ++i)
	{
		for (size_t j = 0; j < res.m_colCount; ++j)
		{
			for (size_t k = 0; k < B.m_rowCount; ++k)
			{
				res.m_values[i * res.m_colCount + j] +=
					A.m_values[i * A.m_colCount + k] * B.m_values[k * B.m_colCount + j];
			}
		}
	}
	return res;
}

template<class T>
std::vector<T> dmMatrix<T>::Multiply(const std::vector<T>& factors)
{
	if (factors.size() != this->m_colCount)
		throw std::exception("Invalid vector size");
	std::vector<T> res(this->m_rowCount);
	for (size_t i = 0; i < this->m_rowCount; ++i)
	{
		T sum(0);
		for (size_t j = 0; j < this->m_colCount; ++j)
		{
			sum += m_values[i * m_colCount + j] * factors[j];
		}
		res[i] = sum;
	}
	return res;
}

template<class T>
const T& dmMatrix<T>::operator()(const size_t i, const size_t j) const
{
	return m_values[i * m_colCount + j];
}

template<class T>
T& dmMatrix<T>::operator()(const size_t i, const size_t j)
{
	return m_values[i * m_colCount + j];
}


template<class T>
dmMatrix<T>::dmMatrix(const size_t rowCount, const size_t colCount)
	:m_values(rowCount * colCount, T()),
	m_rowCount(rowCount),
	m_colCount(colCount)
{
}

template<class T>
dmMatrix<T>::~dmMatrix()
{
}