#pragma once
#include <vector>
#include <algorithm>
#include <immintrin.h>
//!  dmMatrix  class is a dense matrix stored row by row
/*!
  
*/

template<class T>
class dmVectorView
{
public:
	dmVectorView() :
		m_size(0), m_data(nullptr)
	{
	}
	dmVectorView(T* data, const size_t size) :
		m_size(size), m_data(data)
	{
	}
	const T& front() const { return m_data[0]; }
	const T& back() const { return *(m_data + (m_size - 1)); }
	T& front() { return m_data[0]; };
	T& back() { return *(m_data + (m_size - 1)); }
	const size_t size() const { return m_size; }
	const T& operator [](const size_t idx) const { return m_data[idx]; }
	T& operator [](const size_t idx) { return m_data[idx]; }

	void resize(const size_t newSize)
	{
		m_size = newSize;
	}

	void set_view(T* data, const size_t size)
	{
		m_size = size;
		m_data = data;
	}

private:
	size_t m_size;
	T* m_data;
};

template<class T>
struct dmBlock
{
	dmBlock(T* values, const size_t rowCount, const size_t colCount);
	dmBlock();
	dmBlock<T> Multiply(const dmBlock<T>& factor, dmBlock<T>& res);
	const size_t GetColCount()
	{
		return m_colCount;
	}
	const size_t GetRowCount()
	{
		return m_rowCount;
	}
	template <class U>
	void MultiplyInPlace(const std::vector<U>& factors, dmVectorView<U>& res) const
	{
		if (factors.size() != this->m_colCount || res.size() != this->m_rowCount)
			throw std::exception("Invalid vector size");
		for (size_t i = 0; i < this->m_rowCount; ++i)
		{
			T sum(0);
			for (size_t j = 0; j < this->m_colCount; ++j)
			{
				sum += U(m_values[i * m_colCount + j]) * factors[j];
			}
			res[i] = sum;
		}
	}

	template <class U>
	void MultiplyInPlace(const std::vector<U>& factors, std::vector<U>& res) const
	{
		if (factors.size() != this->m_colCount || res.size() != this->m_rowCount)
			throw std::exception("Invalid vector size");
		for (size_t i = 0; i < this->m_rowCount; ++i)
		{
			T sum(0);
			for (size_t j = 0; j < this->m_colCount; ++j)
			{
				sum += U(m_values[i * m_colCount + j]) * factors[j];
			}
			res[i] = sum;
		}
	}
	void set_view(T* data, const size_t rowCount, const size_t colCount)
	{
		m_rowCount = rowCount;
		m_colCount = colCount;
		m_values = data;
	}

	T& operator()(const size_t i, const size_t j)
	{
		return m_values[i * m_colCount + j];
	}
	
	const T& operator()(const size_t i, const size_t j) const
	{
		return m_values[i * m_colCount + j];
	}

	void UploadToMem(T* memory) const;
	size_t m_colCount;
	size_t m_rowCount;
	T* m_values;
};

template<class T>
void dmBlock<T>::UploadToMem(T* memory) const
{
	for (size_t i = 0; i < m_colCount * m_rowCount; ++i)
	{
		memory[i] = m_values[i];
	}
}

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
dmBlock<T>::dmBlock(T* values, const size_t rowCount, const size_t colCount)
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
	template <class U>
	std::vector<U> Multiply(const std::vector<U>& factors) const
	{
		if (factors.size() != this->m_colCount)
			throw std::exception("Invalid vector size");
		std::vector<U> res(this->m_rowCount);
		for (size_t i = 0; i < this->m_rowCount; ++i)
		{
			T sum(0);
			for (size_t j = 0; j < this->m_colCount; ++j)
			{
				sum += U(m_values[i * m_colCount + j]) * factors[j];
			}
			res[i] = sum;
		}
		return res;
	}

	dmMatrix<T> UpperTriangular() const
	{
		if (this->m_rowCount != this->m_colCount)
			throw std::exception("Input matrix must be square!");
		dmMatrix<T> res(m_rowCount, m_colCount);
		for (size_t i = 0; i < m_rowCount; ++i)
		{
			for (size_t j = i; j < m_colCount; ++j)
			{
				res(i, j) = (*this)(i, j);
			}
		}
		return res;
	}

	template<class U>
	dmMatrix<U> Multiply(const dmMatrix<U>& factors) const
	{
		if (this->m_colCount != factors.m_rowCount)
			throw std::exception("Invalid matrix size");

		dmMatrix<U> res(this->m_rowCount, factors.m_colCount);
		const int M = res.m_rowCount;
		const int N = res.m_rowCount;
		const int K = factors.m_rowCount;
		U* C = &res.m_values[0];
		const T* A = this->m_values.data();
		const U* B = factors.m_values.data();
		for (int i = 0; i < M; ++i)
		{
			U* c = C + i * N;
			for (int j = 0; j < N; ++j)
				c[j] = 0;
			for (int k = 0; k < K; ++k)
			{
				const U* b = B + k * N;
				T a = A[i * K + k];
				for (int j = 0; j < N; ++j)
					c[j] += U(a) * b[j];
			}
		}
		return res;
	}

	void Print() const
	{
		for (size_t i = 0; i < this->GetRowCount(); ++i)
		{
			for (size_t j = 0; j < this->GetColCount(); ++j)
			{
				std::cout << (*this)(i, j) << " ";
			}
			std::cout << std::endl;
		}
	}

	size_t GetColCount() const;
	size_t GetRowCount() const;
	void SwapRows(const size_t idx1, const size_t idx2);
	void Transpose();
	void TransposeInPlace();

	template<class U>
	dmMatrix<U> operator * (const dmMatrix<U>& factor) const
	{
		return this->Multiply(factor);
	}

	dmMatrix<T> operator + (const dmMatrix<T>& factor) const
	{
		if (this->m_values.size() != factor.m_values.size())
			throw std::exception("Matrix size should be equal");
		dmMatrix<T> res(factor.GetRowCount(), factor.GetColCount());
		for (size_t i = 0; i < factor.m_values.size(); ++i)
		{
			res.m_values[i] = factor.m_values[i] + this->m_values[i];
		}
		return res;
	}

	dmMatrix<T> operator - (const dmMatrix<T>& factor) const
	{
		if (this->m_values.size() != factor.m_values.size())
			throw std::exception("Matrix size should be equal");
		dmMatrix<T> res(factor.GetRowCount(), factor.GetColCount());
		for (size_t i = 0; i < factor.m_values.size(); ++i)
		{
			res.m_values[i] = this->m_values[i] - factor.m_values[i];
		}
		return res;
	}

	const dmMatrix<T>& operator += (const dmMatrix<T>& factor)
	{
		if (this->m_values.size() != factor.m_values.size())
			throw std::exception("Matrix size should be equal");
		for (size_t i = 0; i < factor.m_values.size(); ++i)
		{
			this->m_values[i] += factor.m_values[i];
		}
		return *this;
	}

	const dmMatrix<T>& operator -= (const dmMatrix<T>& factor)
	{
		if (this->m_values.size() != factor.m_values.size())
			throw std::exception("Matrix size should be equal");
		for (size_t i = 0; i < factor.m_values.size(); ++i)
		{
			this->m_values[i] -= factor.m_values[i];
		}
		return *this;
	}

	const dmMatrix<T>& operator *= (const T factor)
	{
		for (size_t i = 0; i < this->m_values.size(); ++i)
		{
			this->m_values[i] *= factor;
		}
		return *this;
	}

	const dmMatrix<T>& operator /= (const T factor)
	{
		for (size_t i = 0; i < this->m_values.size(); ++i)
		{
			this->m_values[i] /= factor;
		}
		return *this;
	}

	dmMatrix<T> operator / (const T factor) const
	{
		dmMatrix<T> res(*this);
		return res /= factor;
	}

	dmMatrix<T> operator * (const T factor) const
	{
		dmMatrix<T> res(*this);
		return res *= factor;
	}

	template<class U>
	std::vector<U> operator * (const std::vector<U>& factor) const
	{
		return this->Multiply(factor);
	}

	dmMatrix<dmBlock<T>> ToBlocked(const dmMatrix<T>& matr, std::vector<T>& memory, const size_t blockSize);
	void FromBlocked(const dmMatrix<dmBlock<T>>& blocked, dmMatrix<T>& matr);

	std::vector<T> m_values;
	size_t m_rowCount;
	size_t m_colCount;
};

template<class T>
void dmMatrix<T>::FromBlocked(const dmMatrix<dmBlock<T>>& blocked, dmMatrix<T>& matr)
{
	size_t id = 0;
	for (size_t i = 0; i < blocked.m_rowCount; ++i)
	{
		const dmBlock<T>& startBlock = blocked.m_values[i * blocked.m_colCount];
		for(size_t rowIdx = 0; rowIdx < startBlock.m_rowCount; ++rowIdx)
		for (size_t j = 0; j < blocked.m_colCount; ++j)
		{
			const dmBlock<T>& block = blocked.m_values[i * blocked.m_colCount + j];
			for (size_t jl = 0; jl < block.m_colCount; ++jl)
			{
				matr.m_values[id++] = block.m_values[rowIdx * block.m_colCount + jl];
			}
		}
	}
}

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
void dmMatrix<T>::SwapRows(const size_t idx1, const size_t idx2)
{
	for (size_t i = 0; i < this->m_colCount; ++i)
	{
		std::swap(m_values[idx1 * m_colCount + i], m_values[idx2 * m_colCount + i]);
	}
}

template<class T>
void dmMatrix<T>::Transpose()
{
	if (this->GetColCount() == this->GetRowCount())
	{
		TransposeInPlace();
		return;
	}
	dmMatrix<T> old(*this);
	std::swap(m_colCount, m_rowCount);
	for (size_t i = 0; i < old.GetRowCount(); i++)
	{
		for (size_t j = 0; j < old.GetColCount(); j++)
		{
			(*this)(j, i) = old(i, j);
		}
	}
	
}

template<class T>
void dmMatrix<T>::TransposeInPlace()
{
	if (this->GetRowCount() != this->GetColCount())
		throw std::exception("Transpose in place is possible only for homogeunuse matrix");
	T tmpVal;
	for (size_t row = 0; row + 1 < this->GetRowCount(); ++row)
	{
		for (size_t col = row + 1; col < this->GetColCount(); ++col)
		{
			tmpVal = (*this)(row, col);
			(*this)(row, col) = (*this)(col, row);
			(*this)(col, row) = tmpVal;
		}
	}
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