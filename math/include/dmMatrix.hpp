#pragma once
#include <vector>
//!  dmMatrix  class is a dense matrix stored row by row
/*!
  
*/

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
	size_t GetColCount() const;
	size_t GetRowCount() const;
	//dmMatrix<T> operator * (const dmMatrix<T>& factor) const;

private:
	std::vector<T> m_values;
	size_t m_rowCount;
	size_t m_colCount;
};

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