#include "mwDropOutLayer.hpp"

namespace layers
{

template<typename Scalar>
mwDropOutLayer<Scalar>::mwDropOutLayer(
	const mwTensorView<Scalar>& inputShape,
	const Scalar treshold /*= Scalar(0.5)*/,
	const int seed /*= Scalar(3731)*/)
	:mwLayer(inputShape),
	m_engine(seed),
	m_uniformDist(Scalar(0), Scalar(1)),
	m_out(inputShape.RowCount(), inputShape.ColCount(), inputShape.Depth()),
	m_delta(inputShape.RowCount(), inputShape.ColCount(), inputShape.Depth()),
	m_activated(inputShape.RowCount(), inputShape.ColCount(), inputShape.Depth()),
	m_treshold(treshold)
{
}

template<typename Scalar>
Scalar mwDropOutLayer<Scalar>::Treshold() const
{
	return m_treshold;
}

template<typename Scalar>
void mwDropOutLayer<Scalar>::Init()
{
}

template<typename Scalar>
void mwDropOutLayer<Scalar>::SetDeltaToZero()
{
	m_delta.ToView().SetToZero();
}

template<typename Scalar>
const mwTensorView<Scalar> mwDropOutLayer<Scalar>::Input() const
{
	return m_in;
}

template<typename Scalar>
const mwTensorView<Scalar> mwDropOutLayer<Scalar>::GetDerivatives() const
{
	const mwTensorView<Scalar>& tensor = m_delta;
	return tensor;
}

template<typename Scalar>
const mwTensorView<Scalar> mwDropOutLayer<Scalar>::Output() const
{
	const mwTensorView<Scalar>& tensor = m_out;
	return tensor;
}

template<typename Scalar>
void mwDropOutLayer<Scalar>::Backprop(const mwTensorView<Scalar>& nextDelta)
{
	mwVectorView<Scalar> nextDeltaVec = nextDelta.ToVectorView();
	mwVectorView<Scalar> deltaVec = m_delta.ToView().ToVectorView();
	mwVectorView<int> activeVec = m_activated.ToView().ToVectorView();
	for (size_t i = 0; i < deltaVec.size(); ++i)
	{
		deltaVec[i] += activeVec[i] ? nextDeltaVec[i] : Scalar(0);
	}
}

template<typename Scalar>
void mwDropOutLayer<Scalar>::CalcGrads(const mwTensorView<Scalar>& /*nextDelta*/)
{
}

template<typename Scalar>
void mwDropOutLayer<Scalar>::Forward(const mwTensorView<Scalar>& input)
{
	m_in = input;
	m_activated.ToView().SetToZero();
	mwVectorView<Scalar> inputVec = input.ToVectorView();
	mwVectorView<Scalar> outVec = m_out.ToView().ToVectorView();
	mwVectorView<int> activeVec = m_activated.ToView().ToVectorView();
	for (size_t i = 0; i < inputVec.size(); ++i)
	{
		const int active = m_uniformDist(m_engine) <= m_treshold;
		activeVec[i] = active;
		outVec[i] = active ? inputVec[i] : Scalar(0);
	}
}

template<typename Scalar>
void mwDropOutLayer<Scalar>::MapData(Scalar* /*weights*/, Scalar* /*gradient*/, Scalar* /*workSpace*/)
{
}

template<typename Scalar>
size_t mwDropOutLayer<Scalar>::OptimizedParamsCount() const
{
	return 0;
}

template<typename Scalar>
mwLayerType mwDropOutLayer<Scalar>::GetType() const
{
	return mwLayerType::DROP_OUT;
}

template<typename Scalar>
mwTensorView<Scalar> mwDropOutLayer<Scalar>::GetOutShape() const
{
	const mwTensorView<Scalar>& tensor = m_out;
	return tensor;
}

template struct mwDropOutLayer<double>;
template struct mwDropOutLayer<float>;

}