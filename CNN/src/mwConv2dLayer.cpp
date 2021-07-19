#include "mwConv2dLayer.hpp"
#include "mwInitialization.hpp"
#include "mwHeInitialization.hpp"

namespace layers
{

template<typename Scalar>
mwConv2dLayer<Scalar>::mwConv2dLayer(
	const size_t featuresCount,
	const size_t kernel,
	const mwTensorView<Scalar>& inputShape,
	const std::shared_ptr<mwInitialization<Scalar>> init /*= mwHeInitialization()*/)
	: mwLayer(inputShape),
	m_out(inputShape.RowCount() - kernel + 1,
		inputShape.ColCount() - kernel + 1,
		featuresCount),
	m_delta(inputShape.RowCount(), inputShape.ColCount(), inputShape.Depth()),
	m_grads(nullptr, kernel, kernel, featuresCount * inputShape.Depth()),
	m_biasGrads(nullptr, 1, 1, featuresCount),
	m_weights(nullptr, kernel, kernel, featuresCount * inputShape.Depth()),
	m_bias(nullptr, 1, 1, featuresCount),
	m_featuresCount(featuresCount),
	m_kernel(kernel),
	m_init(init)
{
}

template<typename Scalar>
const mwTensorView<Scalar> mwConv2dLayer<Scalar>::GetDerivatives() const
{
	const mwTensorView<Scalar>& tensor = m_delta;
	return tensor;
}

template<typename Scalar>
const mwTensorView<Scalar> mwConv2dLayer<Scalar>::Output() const
{
	const mwTensorView<Scalar>& tensor = m_out;
	return tensor;
}

template<typename Scalar>
void mwConv2dLayer<Scalar>::SetDeltaToZero()
{
	m_delta.ToView().SetToZero();
	m_grads.SetToZero();
	m_biasGrads.SetToZero();
}

template<typename Scalar>
const mwTensorView<Scalar> mwConv2dLayer<Scalar>::Input() const
{
	return m_in;
}


template<typename Scalar>
size_t mwConv2dLayer<Scalar>::Kernel() const { return m_kernel; }

template<typename Scalar>
size_t mwConv2dLayer<Scalar>::FeaturesCount() const { return m_featuresCount; }

template<typename Scalar>
void mwConv2dLayer<Scalar>::Init()
{
	m_init->Init(m_weights.Size(), m_weights);
	m_init->Init(m_bias.Size(), m_bias);
}

template<typename Scalar>
void mwConv2dLayer<Scalar>::Backprop(
	const mwTensorView<Scalar>& nextDelta)
{
	const size_t step = m_kernel - 1;
	mwTensorView<Scalar> outView = m_out.ToView();
	mwTensorView<Scalar> deltaView = m_delta.ToView();
	for (size_t feature = 0; feature < outView.Depth(); ++feature)
	{
		dmMatrixView<Scalar> matrOutDelta = nextDelta(feature);
		for (size_t d = 0; d < deltaView.Depth(); ++d)
		{
			dmMatrixView<Scalar> W = m_weights(feature * m_in.Depth() + d);
			dmMatrixView<Scalar> matrInDelta = deltaView(d);
			for (size_t i = 0; i + step < deltaView.RowCount(); ++i)
			{
				for (size_t j = 0; j + step < deltaView.ColCount(); ++j)
				{
					for (size_t ki = 0; ki < m_weights.RowCount(); ++ki)
					{
						for (size_t kj = 0; kj < m_weights.ColCount(); ++kj)
						{
							matrInDelta(i + ki, j + kj) +=
								matrOutDelta(i, j) * W(ki, kj);
						}
					}
				}
			}
		}
	}
}

template<typename Scalar>
void mwConv2dLayer<Scalar>::CalcGrads(const mwTensorView<Scalar>& nextDelta)
{
	const size_t step = m_kernel - 1;
	mwTensorView<Scalar> outView = m_out.ToView();
	for (size_t feature = 0; feature < outView.Depth(); ++feature)
	{
		Scalar& gradBias = m_biasGrads(feature)(0, 0);
		dmMatrixView<Scalar> matrNextDelta = nextDelta(feature);
		for (size_t d = 0; d < m_in.Depth(); ++d)
		{
			dmMatrixView<Scalar> matrGrads = m_grads(feature * m_in.Depth() + d);
			dmMatrixView<Scalar> matrIn = m_in(d);
			for (size_t i = 0; i + step < m_in.RowCount(); ++i)
			{
				for (size_t j = 0; j + step < m_in.ColCount(); ++j)
				{
					for (size_t ki = 0; ki < m_weights.RowCount(); ++ki)
					{
						for (size_t kj = 0; kj < m_weights.ColCount(); ++kj)
						{
							matrGrads(ki, kj) += matrIn(i + ki, j + kj) * matrNextDelta(i, j);
						}
					}
					gradBias += matrNextDelta(i, j);
				}
			}
		}
	}
}

template<typename Scalar>
void mwConv2dLayer<Scalar>::Forward(const mwTensorView<Scalar>& input)
{
	m_in = input;
	const size_t step = m_kernel - 1;
	mwTensorView<Scalar> outView = m_out.ToView();
	outView.SetToZero();

	for (size_t feature = 0; feature < outView.Depth(); ++feature)
	{
		dmMatrixView<Scalar> matrOut = outView(feature);
		
		Scalar B = m_bias(0, 0, feature);
		for (size_t d = 0; d < input.Depth(); ++d)
		{
			dmMatrixView<Scalar> W = m_weights(feature * input.Depth() + d);
			dmMatrixView<Scalar> matrIn = input(d);
			for (size_t i = 0; i + step < input.RowCount(); ++i)
			{
				for (size_t j = 0; j + step < input.ColCount(); ++j)
				{
					Scalar sum = 0;
					for (size_t ki = 0; ki < W.RowCount(); ++ki)
					{
						for (size_t kj = 0; kj < W.ColCount(); ++kj)
						{
							sum += matrIn(i + ki, j + kj) * W(ki, kj);
						}
					}
					matrOut(i, j) += sum + B;
				}
			}
		}
	}
}

template<typename Scalar>
void mwConv2dLayer<Scalar>::MapData(Scalar* weights, Scalar* gradient)
{
	m_weights.SetView(weights);
	m_bias.SetView(weights + m_weights.Size());
	m_grads.SetView(gradient);
	m_biasGrads.SetView(gradient + m_grads.Size());
}

template<typename Scalar>
size_t mwConv2dLayer<Scalar>::OptimizedParamsCount() const
{
	return m_weights.Size() + m_bias.Size();
}

template<typename Scalar>
mwLayerType mwConv2dLayer<Scalar>::GetType() const
{
	return mwLayerType::CONVOLUTION;
}

template<typename Scalar>
mwTensorView<Scalar> mwConv2dLayer<Scalar>::GetOutShape() const
{
	const mwTensorView<Scalar>& tensor = m_out;
	return tensor;
}


template struct mwConv2dLayer<double>;
template struct mwConv2dLayer<float>;

}