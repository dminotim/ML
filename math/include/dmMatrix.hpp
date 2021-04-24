#pragma once
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
	//dmMatrix<T> operator * (const dmMatrix<T>& factor) const;

private:
	std::vector<T> m_values;
	size_t m_rowCount;
	size_t m_colCount;
};

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