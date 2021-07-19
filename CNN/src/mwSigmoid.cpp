#include "mwSigmoid.hpp"

namespace layers
{
template<typename Scalar>
mwSigmoid<Scalar>::mwSigmoid(
	const mwTensorView<Scalar>& inputShape)
	:mwLayer(inputShape),
	m_out(inputShape.RowCount(), inputShape.ColCount(), inputShape.Depth()),
	m_delta(inputShape.RowCount(), inputShape.ColCount(), inputShape.Depth())
{
	Init();
}

template<typename Scalar>
void mwSigmoid<Scalar>::Init()
{
}

template<typename Scalar>
void mwSigmoid<Scalar>::SetDeltaToZero()
{
	m_delta.ToView().SetToZero();
}

template<typename Scalar>
const mwTensorView<Scalar> mwSigmoid<Scalar>::Input() const
{
	return m_in;
}

template<typename Scalar>
const mwTensorView<Scalar> mwSigmoid<Scalar>::GetDerivatives() const
{
	const mwTensorView<Scalar> tensor = m_delta;
	return tensor;
}

template<typename Scalar>
const mwTensorView<Scalar> mwSigmoid<Scalar>::Output() const
{
	const mwTensorView<Scalar> tensor = m_out;
	return tensor;
}

template<typename Scalar>
void mwSigmoid<Scalar>::Backprop(const mwTensorView<Scalar>& nextDelta)
{
	mwVectorView<Scalar> deltaVec = m_delta.ToView().ToVectorView();
	mwVectorView<Scalar> nextDeltaVec = nextDelta.ToVectorView();
	mwVectorView<Scalar> inputVec = m_in.ToVectorView();
	for (size_t i = 0; i < nextDeltaVec.size(); ++i)
	{
		deltaVec[i] += inputVec[i] * (1.0f - inputVec[i]);
	}
}

template<typename Scalar>
void mwSigmoid<Scalar>::CalcGrads(const mwTensorView<Scalar>& /*nextDelta*/)
{
}

template<typename Scalar>
void mwSigmoid<Scalar>::Forward(const mwTensorView<Scalar>& input)
{
	m_in = input;
	mwVectorView<Scalar> inputVec = input.ToVectorView();
	mwVectorView<Scalar> outVec = m_out.ToView().ToVectorView();
	for (size_t i = 0; i < inputVec.size(); ++i)
	{
		outVec[i] = Scalar(1) / (Scalar(1) + std::exp(-Scalar(1) * inputVec[i]));
	}
}

template<typename Scalar>
void mwSigmoid<Scalar>::MapData(Scalar* /*weights*/, Scalar* /*gradient*/)
{
}

template<typename Scalar>
size_t mwSigmoid<Scalar>::OptimizedParamsCount() const
{
	return 0;
}

template<typename Scalar>
mwLayerType mwSigmoid<Scalar>::GetType() const
{
	return mwLayerType::SIGMOID;
}

template<typename Scalar>
mwTensorView<Scalar> mwSigmoid<Scalar>::GetOutShape() const
{
	const mwTensorView<Scalar>& tensor = m_out;
	return tensor;
}

template struct mwSigmoid<double>;
template struct mwSigmoid<float>;

}