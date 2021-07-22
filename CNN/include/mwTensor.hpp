#pragma  once
#include <vector>

template<class T>
class mwVectorView
{
public:
	mwVectorView() :
		m_size(0), m_data(nullptr)
	{
	}
	mwVectorView(T* data, const size_t size) :
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

	void SetView(T* data, const size_t size)
	{
		m_size = size;
		m_data = data;
	}

private:
	size_t m_size;
	T* m_data;
};

void gemm_nn(int M, int N, int K,
	float* A, int lda,
	float* B, int ldb,
	float* C, int ldc);

void gemm_nn(int M, int N, int K,
	double* A, int lda,
	double* B, int ldb,
	double* C, int ldc);

template<class T>
struct dmMatrixView
{
	dmMatrixView(T* values, const size_t rowCount, const size_t colCount)
		:m_values(values), m_colCount(colCount), m_rowCount(rowCount)
	{
	}

	dmMatrixView()
		:m_values(nullptr), m_colCount(0), m_rowCount(0)
	{
	}

	const size_t ColCount()
	{
		return m_colCount;
	}

	const size_t RowCount()
	{
		return m_rowCount;
	}

	template <class U>
	void MultiplyInPlace(const mwVectorView<U>& factors, mwVectorView<U> res) const
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

	void Multiply(const dmMatrixView<T>& factors, dmMatrixView<T>& res)
	{
		if (this->m_colCount != factors.m_rowCount)
			throw std::exception("Invalid matrix size");

		const dmMatrixView<T>& A = *this;
		const dmMatrixView<T>& B = factors;
		gemm_nn(A.m_rowCount,
			factors.m_colCount,
			factors.m_rowCount,
			A.m_values,
			A.m_colCount,
			B.m_values,
			B.m_colCount,
			res.m_values,
			res.m_colCount);
		/*for (size_t i = 0; i < res.m_rowCount; ++i)
		{
			for (size_t j = 0; j < res.m_colCount; ++j)
			{
				for (size_t k = 0; k < B.m_rowCount; ++k)
				{
					res.m_values[i * res.m_colCount + j] +=
						A.m_values[i * A.m_colCount + k] * B.m_values[k * B.m_colCount + j];
				}
			}
		}*/
	}

	void SetView(T* data, const size_t rowCount, const size_t colCount)
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

	size_t m_colCount;
	size_t m_rowCount;
	T* m_values;
};

template<typename Scalar>
class mwTensor;

template<typename Scalar>
struct mwTensorView
{

	mwTensorView(Scalar* values, const size_t rowCount, const size_t colCount, const size_t depth)
		: m_valuesPtr(values), m_rowCount(rowCount), m_colCount(colCount), m_depth(depth)
	{
	}
	mwTensorView()
		: m_valuesPtr(nullptr), m_rowCount(0), m_colCount(0), m_depth(0)
	{
	}


	size_t RowCount() const { return m_rowCount; }
	void RowCount(size_t val) { m_rowCount = val; }

	size_t ColCount() const { return m_colCount; }
	void ColCount(size_t val) { m_colCount = val; }

	size_t Depth() const { return m_depth; }
	void Depth(size_t val) { m_depth = val; }

	Scalar* Values() const { return m_valuesPtr; }
	void Values(Scalar* val) { m_valuesPtr = val; }

	const size_t Size() const
	{
		return m_rowCount * m_colCount * m_depth;
	}

	mwTensorView<Scalar> GetSubTensorByDepth(const size_t startDIncluded, const size_t endDNotIncluded) const
	{
		mwTensorView<Scalar> res(m_valuesPtr + startDIncluded * m_rowCount * m_colCount,
			m_rowCount, m_colCount, endDNotIncluded - startDIncluded);
		return res;
	}

	mwVectorView<Scalar> ToVectorView()
	{
		return mwVectorView<Scalar>(m_valuesPtr, Size());
	}

	const mwVectorView<Scalar> ToVectorView() const
	{
		return mwVectorView<Scalar>(m_valuesPtr, Size());
	}

	void SetToZero()
	{
		for (size_t i = 0; i < Size(); ++i)
		{
			m_valuesPtr[i] = Scalar(0);
		}
	}

	dmMatrixView<Scalar> operator ()(const size_t depthIdx)
	{
		return dmMatrixView<Scalar>(&m_valuesPtr[depthIdx * m_rowCount * m_colCount], m_rowCount, m_colCount);
	}

	Scalar& operator ()(const size_t row, const size_t col, const size_t d)
	{
		return m_valuesPtr[d * m_rowCount * m_colCount + row * m_colCount + col];
	}

	const dmMatrixView<Scalar> operator ()(const size_t depthIdx) const
	{
		return dmMatrixView<Scalar>(&m_valuesPtr[depthIdx * m_rowCount * m_colCount], m_rowCount, m_colCount);
	}

	const Scalar& operator ()(const size_t row, const size_t col, const size_t d) const
	{
		return  m_valuesPtr[d * m_rowCount * m_colCount + row * m_colCount + col];
	}

	void SetView(Scalar* data)
	{
		m_valuesPtr = data;
	}

	mwTensor<Scalar> DeepCopy()
	{
		const size_t size = this->m_colCount * this->m_rowCount * this->m_depth;
		mwTensor<Scalar> ten(this->m_rowCount, this->m_colCount, this->m_depth);

		for (size_t i = 0; i < size; ++i)
		{
			ten.m_values[i] = m_valuesPtr[i];
		}
		return ten;
	}

	void TarnsposeZeroDepth()
	{
		const size_t size = this->m_colCount * this->m_rowCount;
		std::vector<Scalar> oldVal(size);
		for (size_t i = 0; i < size; ++i)
		{
			oldVal[i] = m_valuesPtr[i];
		}
		dmMatrixView<Scalar> old(&oldVal[0], m_rowCount, m_colCount);
		std::swap(m_colCount, m_rowCount);
		dmMatrixView<Scalar> self = (*this)(0);
		for (size_t i = 0; i < old.RowCount(); i++)
		{
			for (size_t j = 0; j < old.ColCount(); j++)
			{
				self(j, i) = old(i, j);
			}
		}
	}

	void MakeTransposedZeroDepth(mwTensor<Scalar>& res)
	{
		res = mwTensor<Scalar>(this->m_colCount, this->m_rowCount, 1);
		mwTensorView<Scalar> resView = res.ToView();
		dmMatrixView<Scalar> resViewM = resView(0);
		dmMatrixView<Scalar> self = (*this)(0);
		for (size_t i = 0; i < self.RowCount(); i++)
		{
			for (size_t j = 0; j < self.ColCount(); j++)
			{
				resViewM(j, i) = self(i, j);
			}
		}
	}



	Scalar* m_valuesPtr;
protected:
	size_t m_rowCount;
	size_t m_colCount;
	size_t m_depth;

};

template<typename Scalar>
struct mwTensor
{
	mwTensor()
	{

	}
	mwTensor(const size_t rowCount, const size_t colCount, const size_t depth)
		: m_rowCount(rowCount), m_colCount(colCount), m_depth(depth)
	{
		m_values.resize(m_colCount * m_rowCount * m_depth, Scalar(0));
	}

	operator mwTensorView<Scalar>() const {
		return mwTensorView<Scalar>((Scalar*)m_values.data(), m_rowCount, m_colCount, m_depth);
	}

	mwTensorView<Scalar> ToView() const {
		return mwTensorView<Scalar>(
			(Scalar*)m_values.data(),
			m_rowCount,
			m_colCount,
			m_depth);
	}
	size_t m_rowCount;
	size_t m_colCount;
	size_t m_depth;
	std::vector<Scalar> m_values;
};