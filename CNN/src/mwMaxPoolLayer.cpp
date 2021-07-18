#include "StdAfx.h"
#include "mwMaxPoolLayer.hpp"
#include <limits>

namespace layers
{

template<typename Scalar>
mwMaxPoolLayer<Scalar>::mwMaxPoolLayer(
	const size_t kernel, const mwTensorView<Scalar>& inputShape)
	:mwLayer(inputShape),
	m_out(inputShape.RowCount() / kernel, inputShape.ColCount() / kernel, inputShape.Depth()),
	m_delta(inputShape.RowCount(), inputShape.ColCount(), inputShape.Depth()),
	m_kernel(kernel)
{
}

template<typename Scalar>
size_t mwMaxPoolLayer<Scalar>::Kernel() const
{
	return m_kernel;
}

template<typename Scalar>
void mwMaxPoolLayer<Scalar>::Init()
{
}

template<typename Scalar>
const mwTensorView<Scalar> mwMaxPoolLayer<Scalar>::GetDerivatives() const
{
	const mwTensorView<Scalar>& tensor = m_delta;
	return tensor;
}

template<typename Scalar>
const mwTensorView<Scalar> mwMaxPoolLayer<Scalar>::Output() const
{
	const mwTensorView<Scalar>& tensor = m_out;
	return tensor;
}

template<typename Scalar>
void mwMaxPoolLayer<Scalar>::SetDeltaToZero()
{
	m_delta.ToView().SetToZero();
}

template<typename Scalar>
const mwTensorView<Scalar> mwMaxPoolLayer<Scalar>::Input() const
{
	return m_in;
}

template<typename Scalar>
void mwMaxPoolLayer<Scalar>::Backprop(const mwTensorView<Scalar>& nextDelta)
{
	const size_t step = m_kernel - 1;
	mwTensorView<Scalar> outView = m_out.ToView();
	mwTensorView<Scalar> deltaView = m_delta.ToView();
	for (size_t d = 0; d < m_in.Depth(); ++d)
	{
		dmMatrixView<Scalar> matrIn = m_in(d);
		dmMatrixView<Scalar> matrOut = outView(d);
		dmMatrixView<Scalar> matrDelta = deltaView(d);
		dmMatrixView<Scalar> matrnextDelta = nextDelta(d);
		for (size_t i = 0; i + step < matrIn.RowCount(); i += m_kernel)
		{
			for (size_t j = 0; j + step < matrIn.ColCount(); j += m_kernel)
			{
				for (size_t ki = 0; ki < m_kernel; ++ki)
				{
					for (size_t kj = 0; kj < m_kernel; ++kj)
					{
						if (matrIn(i + ki, j + kj) == matrOut(i / m_kernel, j / m_kernel))
						{
							matrDelta(i + ki, j + kj) += matrnextDelta(i / m_kernel, j / m_kernel);
						}
					}
				}
			}
		}
	}
}

template<typename Scalar>
void mwMaxPoolLayer<Scalar>::CalcGrads(const mwTensorView<Scalar>& /*nextDelta*/)
{
}

template<typename Scalar>
void mwMaxPoolLayer<Scalar>::Forward(const mwTensorView<Scalar>& input)
{
	m_in = input;
	const size_t step = m_kernel - 1;
	m_out.ToView().SetToZero();
	for (size_t d = 0; d < input.Depth(); ++d)
	{
		dmMatrixView<Scalar> matrIn = input(d);
		dmMatrixView<Scalar> matrOut = m_out.ToView()(d);
		for (size_t i = 0; i + step < matrIn.RowCount(); i += m_kernel)
		{
			for (size_t j = 0; j + step < matrIn.ColCount(); j += m_kernel)
			{
				Scalar maxVal = std::numeric_limits<Scalar>::min();
				for (size_t ki = 0; ki < m_kernel; ++ki)
				{
					for (size_t kj = 0; kj < m_kernel; ++kj)
					{
						maxVal = std::max(matrIn(i + ki, j + kj), maxVal);
					}
				}
				matrOut(i / m_kernel, j / m_kernel) = maxVal;
			}
		}
	}
}

template<typename Scalar>
void mwMaxPoolLayer<Scalar>::MapData(Scalar* /*weights*/, Scalar* /*gradient*/)
{
}

template<typename Scalar>
size_t mwMaxPoolLayer<Scalar>::OptimizedParamsCount() const
{
	return 0;
}

template<typename Scalar>
mwLayerType mwMaxPoolLayer<Scalar>::GetType() const
{
	return mwLayerType::MAX_POOL;
}

template<typename Scalar>
mwTensorView<Scalar> mwMaxPoolLayer<Scalar>::GetOutShape() const
{
	const mwTensorView<Scalar>& tensor = m_out;
	return tensor;
}

template struct mwMaxPoolLayer<double>;
template struct mwMaxPoolLayer<float>;

}