#include "mwReluLayer.hpp"

namespace layers
{
template<typename Scalar>
mwReluLayer<Scalar>::mwReluLayer(
	const mwTensorView<Scalar>& inputShape,
	const Scalar leakEpsilon /*= 0*/)
	:mwLayer(inputShape),
	m_out(inputShape.RowCount(), inputShape.ColCount(), inputShape.Depth()),
	m_delta(inputShape.RowCount(), inputShape.ColCount(), inputShape.Depth()),
	m_leakEpsilon(leakEpsilon)
{
	Init();
}

template<typename Scalar>
Scalar mwReluLayer<Scalar>::LeakEpsilon() const
{
	return m_leakEpsilon;
}

template<typename Scalar>
void mwReluLayer<Scalar>::Init()
{
}

template<typename Scalar>
void mwReluLayer<Scalar>::SetDeltaToZero()
{
	m_delta.ToView().SetToZero();
}

template<typename Scalar>
const mwTensorView<Scalar> mwReluLayer<Scalar>::Input() const
{
	return m_in;
}

template<typename Scalar>
const mwTensorView<Scalar> mwReluLayer<Scalar>::GetDerivatives() const
{
	const mwTensorView<Scalar> tensor = m_delta;
	return tensor;
}

template<typename Scalar>
const mwTensorView<Scalar> mwReluLayer<Scalar>::Output() const
{
	const mwTensorView<Scalar> tensor = m_out;
	return tensor;
}

template<typename Scalar>
void mwReluLayer<Scalar>::Backprop(const mwTensorView<Scalar>& nextDelta)
{
	mwVectorView<Scalar> deltaVec = m_delta.ToView().ToVectorView();
	mwVectorView<Scalar> nextDeltaVec = nextDelta.ToVectorView();
	mwVectorView<Scalar> inputVec = m_in.ToVectorView();
	for (size_t i = 0; i < nextDeltaVec.size(); ++i)
	{
		deltaVec[i] += ((inputVec[i] < 0) ? m_leakEpsilon : Scalar(1)) * nextDeltaVec[i];
	}
}

template<typename Scalar>
void mwReluLayer<Scalar>::CalcGrads(const mwTensorView<Scalar>& /*nextDelta*/)
{
}

template<typename Scalar>
void mwReluLayer<Scalar>::Forward(const mwTensorView<Scalar>& input)
{
	m_in = input;
	mwVectorView<Scalar> inputVec = input.ToVectorView();
	mwVectorView<Scalar> outVec = m_out.ToView().ToVectorView();
	for (size_t i = 0; i < inputVec.size(); ++i)
	{
		outVec[i] = ((inputVec[i] < 0) ? m_leakEpsilon : Scalar(1)) * inputVec[i];
	}
}

template<typename Scalar>
void mwReluLayer<Scalar>::MapData(Scalar* /*weights*/, Scalar* /*gradient*/)
{
}

template<typename Scalar>
size_t mwReluLayer<Scalar>::OptimizedParamsCount() const
{
	return 0;
}

template<typename Scalar>
mwLayerType mwReluLayer<Scalar>::GetType() const
{
	return mwLayerType::RELU;
}

template<typename Scalar>
mwTensorView<Scalar> mwReluLayer<Scalar>::GetOutShape() const
{
	const mwTensorView<Scalar>& tensor = m_out;
	return tensor;
}

template struct mwReluLayer<double>;
template struct mwReluLayer<float>;

}