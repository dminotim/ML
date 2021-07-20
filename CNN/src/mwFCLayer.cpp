#include "mwFCLayer.hpp"

namespace layers
{
template<typename Scalar>
mwFCLayer<Scalar>::mwFCLayer(
	const size_t outSize,
	const mwTensorView<Scalar>& inputShape,
	const std::shared_ptr<mwInitialization<Scalar>> init
		/*= std::make_shared<mwHeInitialization<Scalar>>()*/)
	:mwLayer(inputShape),
	m_out(1, 1, outSize),
	m_delta(inputShape.RowCount(), inputShape.ColCount(), inputShape.Depth()),
	m_grads(nullptr, outSize, inputShape.Size(), 1),
	m_biasGrads(nullptr, 1, 1, outSize),
	m_weights(nullptr, outSize, inputShape.Size(), 1),
	m_bias(nullptr, 1, 1, outSize),
	m_init(init)
{
}

template<typename Scalar>
void mwFCLayer<Scalar>::Init()
{
	m_init->Init(m_weights.ColCount(), m_weights);
	m_init->Init(m_bias.Size(), m_bias);
}


template<typename Scalar>
void mwFCLayer<Scalar>::SetDeltaToZero()
{
	m_delta.ToView().SetToZero();
	m_grads.SetToZero();
	m_biasGrads.SetToZero();
}

template<typename Scalar>
const mwTensorView<Scalar> mwFCLayer<Scalar>::Input() const
{
	return m_in;
}

template<typename Scalar>
const mwTensorView<Scalar> mwFCLayer<Scalar>::GetDerivatives() const
{
	const mwTensorView<Scalar> tensor = m_delta;
	return tensor;
}

template<typename Scalar>
const mwTensorView<Scalar> mwFCLayer<Scalar>::Output() const
{
	const mwTensorView<Scalar> tensor = m_out;
	return tensor;
}

template<typename Scalar>
void mwFCLayer<Scalar>::Backprop(
	const mwTensorView<Scalar>& nextDelta)
{
	mwVectorView<Scalar> outVec = m_out.ToView().ToVectorView();
	mwVectorView<Scalar> deltaVec = m_delta.ToView().ToVectorView();
	mwVectorView<Scalar> nextDeltaVec = nextDelta.ToVectorView();
	dmMatrixView<Scalar> W = m_weights(0);
	for (size_t i = 0; i < deltaVec.size(); ++i)
	{
		for (size_t j = 0; j < outVec.size(); ++j)
		{
			deltaVec[i] += W(j, i) * nextDeltaVec[j];
		}
	}
}

template<typename Scalar>
void mwFCLayer<Scalar>::CalcGrads(const mwTensorView<Scalar>& nextDelta)
{
	mwVectorView<Scalar> outVec = m_out.ToView().ToVectorView();
	mwVectorView<Scalar> inVec = m_in.ToVectorView();
	mwVectorView<Scalar> nextDeltaVec = nextDelta.ToVectorView();
	dmMatrixView<Scalar> G = m_grads(0);
	mwVectorView<Scalar> BG = m_biasGrads.ToVectorView();
	for (size_t j = 0; j < outVec.size(); ++j)
	{
		for (size_t i = 0; i < inVec.size(); ++i)
		{
			G(j, i) += inVec[i] * nextDeltaVec[j];
		}
		BG[j] += nextDeltaVec[j];
	}
}

template<typename Scalar>
void mwFCLayer<Scalar>::Forward(const mwTensorView<Scalar>& input)
{
	m_in = input;
	mwVectorView<Scalar> outVec = m_out.ToView().ToVectorView();
	mwVectorView<Scalar> biasVec = m_bias.ToVectorView();
	m_weights(0).MultiplyInPlace(input.ToVectorView(), outVec);
	for (size_t i = 0; i < outVec.size(); ++i)
	{
		outVec[i] += biasVec[i];
	}
}

template<typename Scalar>
void mwFCLayer<Scalar>::MapData(Scalar* weights, Scalar* gradient, Scalar* wokSpace)
{
	m_weights.SetView(weights);
	m_bias.SetView(weights + m_weights.Size());
	m_grads.SetView(gradient);
	m_biasGrads.SetView(gradient + m_grads.Size());
	m_workSpace.SetView(wokSpace);
}

template<typename Scalar>
size_t mwFCLayer<Scalar>::OptimizedParamsCount() const
{
	return m_weights.Size() + m_bias.Size();
}

template<typename Scalar>
mwLayerType mwFCLayer<Scalar>::GetType() const
{
	return mwLayerType::FULLY_CONNECTED;
}

template<typename Scalar>
mwTensorView<Scalar> mwFCLayer<Scalar>::GetOutShape() const
{
	const mwTensorView<Scalar> tensor = m_out;
	return tensor;
}


template struct mwFCLayer<double>;
template struct mwFCLayer<float>;

}