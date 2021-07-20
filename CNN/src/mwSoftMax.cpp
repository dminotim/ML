#include "mwSoftMax.hpp"

namespace layers
{

template<typename Scalar>
mwSoftMax<Scalar>::mwSoftMax(const mwTensorView<Scalar>& inputShape)
	:mwLayer(inputShape),
	m_out(inputShape.RowCount(), inputShape.ColCount(), inputShape.Depth()),
	m_delta(inputShape.RowCount(), inputShape.ColCount(), inputShape.Depth())
{
}

template<typename Scalar>
void mwSoftMax<Scalar>::SetDeltaToZero()
{
	m_delta.ToView().SetToZero();
}

template<typename Scalar>
const mwTensorView<Scalar> mwSoftMax<Scalar>::Input() const
{
	return m_in;
}

template<typename Scalar>
void mwSoftMax<Scalar>::Init()
{
}

template<typename Scalar>
const mwTensorView<Scalar> mwSoftMax<Scalar>::GetDerivatives() const
{
	return m_delta.ToView();
}

template<typename Scalar>
const mwTensorView<Scalar> mwSoftMax<Scalar>::Output() const
{
	return m_out.ToView();
}

template<typename Scalar>
void mwSoftMax<Scalar>::Backprop(const mwTensorView<Scalar>& nextDelta)
{
	mwVectorView<Scalar> deltaVec = m_delta.ToView().ToVectorView();
	mwVectorView<Scalar> nextDeltaVec = nextDelta.ToVectorView();
	mwVectorView<Scalar> inputVec = m_in.ToVectorView();
	mwVectorView<Scalar> outVec = m_out.ToView().ToVectorView();
	for (size_t i = 0; i < deltaVec.size(); ++i)
	{
		for (size_t j = 0; j < nextDeltaVec.size(); ++j)
		{
			if (i == j)
			{
				deltaVec[i] += outVec[i] * (Scalar(1) - outVec[i]) * nextDeltaVec[j];
			}
			else
			{
				deltaVec[i] -= outVec[i] * outVec[j] * nextDeltaVec[j];
			}
		}
	}
}

template<typename Scalar>
void mwSoftMax<Scalar>::CalcGrads(const mwTensorView<Scalar>& /*nextDelta*/)
{
}

template<typename Scalar>
void mwSoftMax<Scalar>::Forward(const mwTensorView<Scalar>& input)
{
	m_in = input;
	mwVectorView<Scalar> outVec = m_out.ToView().ToVectorView();
	mwVectorView<Scalar> inVec = m_in.ToVectorView();
	Scalar maxV = std::numeric_limits<Scalar>::min();
	for (size_t i = 0; i < outVec.size(); ++i)
	{
		maxV = std::max(maxV, inVec[i]);
	}
	Scalar sum = 0;
	for (size_t i = 0; i < outVec.size(); ++i)
	{
		outVec[i] = std::exp(inVec[i] - maxV);
		sum += outVec[i];
	}
	for (size_t i = 0; i < outVec.size(); ++i)
	{
		outVec[i] /= sum;
	}
}

template<typename Scalar>
void mwSoftMax<Scalar>::MapData(Scalar* /*weights*/, Scalar* /*gradient*/, Scalar* /*wokSpace*/)
{
}

template<typename Scalar>
size_t mwSoftMax<Scalar>::OptimizedParamsCount() const
{
	return 0;
}

template<typename Scalar>
mwLayerType mwSoftMax<Scalar>::GetType() const
{
	return mwLayerType::SOFTMAX;
}

template<typename Scalar>
mwTensorView<Scalar> mwSoftMax<Scalar>::GetOutShape() const
{
	return m_out.ToView();
}
template struct mwSoftMax<double>;
template struct mwSoftMax<float>;
}