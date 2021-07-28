#include "mwConv2dLayer.hpp"
#include "mwInitialization.hpp"
#include "mwHeInitialization.hpp"
#include "mwCNNUtils.hpp"
#include <iostream>

namespace layers
{

template<typename Scalar>
mwConv2dLayer<Scalar>::mwConv2dLayer(
	const size_t featuresCount,
	const size_t kernel,
	const mwPaddingType ptype,
	const mwTensorView<Scalar>& inputShape,
	const std::shared_ptr<mwInitialization<Scalar>> init /*= mwHeInitialization()*/)
	: mwLayer(inputShape),
	m_out((ptype == mwPaddingType::SAME)
			? (inputShape.RowCount()) : (inputShape.RowCount() - kernel + 1),
		(ptype == mwPaddingType::SAME)
		? (inputShape.ColCount()) : (inputShape.ColCount() - kernel + 1),
		featuresCount),
	m_delta(inputShape.RowCount(), inputShape.ColCount(), inputShape.Depth()),
	m_grads(nullptr, kernel, kernel, featuresCount * inputShape.Depth()),
	m_biasGrads(nullptr, 1, 1, featuresCount),
	m_weights(nullptr, kernel, kernel, featuresCount * inputShape.Depth()),
	m_bias(nullptr, 1, 1, featuresCount),
	m_featuresCount(featuresCount),
	m_kernel(kernel),
	m_init(init),
	m_workSpace(nullptr, inputShape.Depth() * kernel * kernel, m_out.ToView().RowCount() * m_out.ToView().ColCount(), 1),
	m_padType(ptype)
{
}

template<typename Scalar>
const mwTensorView<Scalar> mwConv2dLayer<Scalar>::GetDerivatives() const
{
	const mwTensorView<Scalar>& tensor = m_delta;
	return tensor;
}

template<typename Scalar>
const mwPaddingType mwConv2dLayer<Scalar>::PadType() const {
	return m_padType;
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
	std::cout << "Conv2 layer " << m_weights.Size() + m_bias.Size() << std::endl;
	m_init->Init(m_weights.Size(), m_weights);
	m_init->Init(m_bias.Size(), m_bias);
}

template<typename Scalar>
void mwConv2dLayer<Scalar>::Backprop(
	const mwTensorView<Scalar>& nextDelta)
{
	//const size_t step = m_kernel - 1;

	const size_t padding = m_padType == mwPaddingType::SAME ? m_kernel / 2 : 0;
	mwTensorView<Scalar> workSpace(m_workSpace.m_valuesPtr,
		m_in.Depth() * m_kernel * m_kernel, nextDelta.RowCount() * nextDelta.ColCount(), 1);
	mwTensorView<Scalar> weight(m_weights.m_valuesPtr, nextDelta.Depth(), workSpace.RowCount(), 1);
	mwTensor<Scalar> transW;
	weight.MakeTransposedZeroDepth(transW);
	weight = transW.ToView();
	mwTensorView<Scalar> deltas(nextDelta.m_valuesPtr, 
		nextDelta.Depth(), workSpace.ColCount(), 1);
	workSpace.SetToZero();
	weight(0).Multiply(deltas(0), workSpace(0));
	mwTensorView<Scalar> deltaView = m_delta.ToView();
	mwCNNUtils::FromColumnImage(workSpace, m_kernel, padding, deltaView);



	//mwTensorView<Scalar> outView = m_out.ToView();
	//mwTensorView<Scalar> deltaView = m_delta.ToView();
	//for (size_t feature = 0; feature < outView.Depth(); ++feature)
	//{
	//	dmMatrixView<Scalar> matrOutDelta = nextDelta(feature);
	//	for (size_t d = 0; d < deltaView.Depth(); ++d)
	//	{
	//		dmMatrixView<Scalar> W = m_weights(feature * m_in.Depth() + d);
	//		dmMatrixView<Scalar> matrInDelta = deltaView(d);
	//		for (size_t i = 0; i + step < deltaView.RowCount(); ++i)
	//		{
	//			for (size_t j = 0; j + step < deltaView.ColCount(); ++j)
	//			{
	//				for (size_t ki = 0; ki < m_weights.RowCount(); ++ki)
	//				{
	//					for (size_t kj = 0; kj < m_weights.ColCount(); ++kj)
	//					{
	//						matrInDelta(i + ki, j + kj) +=
	//							matrOutDelta(i, j) * W(ki, kj);
	//					}
	//				}
	//			}
	//		}
	//	}
	//}
}

template<typename Scalar>
void mwConv2dLayer<Scalar>::CalcGrads(const mwTensorView<Scalar>& nextDelta)
{
	const size_t padding = m_padType == mwPaddingType::SAME ? m_kernel / 2 : 0;
	mwTensorView<Scalar> workSpace(m_workSpace.m_valuesPtr,
		m_in.Depth() * m_kernel * m_kernel, nextDelta.RowCount() * nextDelta.ColCount(), 1);

	mwTensorView<Scalar> ww(m_weights.m_valuesPtr, m_out.m_depth, workSpace.RowCount(), 1);

	mwCNNUtils::ToRowImage(m_in, m_kernel, padding, workSpace);

	mwTensorView<Scalar> delats(nextDelta.m_valuesPtr, nextDelta.Depth(), workSpace.RowCount(), 1);
	mwTensorView<Scalar> gards(m_grads.m_valuesPtr, nextDelta.Depth(), workSpace.ColCount(), 1);
	delats(0).Multiply(workSpace(0), gards(0));

	//for (size_t feature = 0; feature < nextDelta.Depth(); ++feature)
	//{
	//	dmMatrixView<Scalar> matrNextDelta = nextDelta(feature);
	//	for (size_t d = 0; d < m_in.Depth(); ++d)
	//	{
	//		dmMatrixView<Scalar> matrGrads = m_grads(feature * m_in.Depth() + d);
	//		dmMatrixView<Scalar> matrIn = m_in(d);
	//		for (size_t i = 0; i + step < m_in.RowCount(); ++i)
	//		{
	//			for (size_t j = 0; j + step < m_in.ColCount(); ++j)
	//			{
	//				for (size_t ki = 0; ki < m_weights.RowCount(); ++ki)
	//				{
	//					for (size_t kj = 0; kj < m_weights.ColCount(); ++kj)
	//					{
	//						matrGrads(ki, kj) += matrIn(i + ki, j + kj) * matrNextDelta(i, j);
	//					}
	//				}
	//			}
	//		}
	//	}
	//}

	//for (size_t i = 0; i < g.m_values.size(); ++i)
	//{
	//	if(std::fabs(m_grads.m_valuesPtr[i] * 2 - g.m_values[i]) > Scalar(1e-4)
	//		&& std::fabs(m_grads.m_valuesPtr[i] - g.m_values[i]) > Scalar(1e-4))
	//	std::cout << g.m_values[i] << " " << m_grads.m_valuesPtr[i] << std::endl;
	//}

	for (size_t feature = 0; feature < nextDelta.Depth(); ++feature)
	{
		Scalar& gradBias = m_biasGrads(feature)(0, 0);
		dmMatrixView<Scalar> matrNextDelta = nextDelta(feature);
		for (size_t i = 0; i < matrNextDelta.RowCount(); ++i)
		{
			for (size_t j = 0; j < matrNextDelta.ColCount(); ++j)
			{
				gradBias += matrNextDelta(i, j);
			}
		}
	}
}

template<typename Scalar>
void mwConv2dLayer<Scalar>::Forward(const mwTensorView<Scalar>& input)
{
	const size_t padding = m_padType == mwPaddingType::SAME ? m_kernel / 2 : 0;
	m_in = input;
	mwTensorView<Scalar> outView = m_out.ToView();
	outView.SetToZero();
	mwTensorView<Scalar> workSpace(m_workSpace.m_valuesPtr,
		m_in.Depth() * m_kernel * m_kernel, outView.RowCount() * outView.ColCount(), 1);
	mwCNNUtils::ToColumnImage(m_in, m_kernel, padding, workSpace);
	mwTensorView<Scalar> ww(m_weights.m_valuesPtr, m_out.m_depth, workSpace.RowCount(), 1);
	mwTensorView<Scalar> outWW(outView.m_valuesPtr, ww.RowCount(), workSpace.ColCount(), 1);
	ww(0).Multiply(workSpace(0), outWW(0));

	/*for (size_t feature = 0; feature < outView.Depth(); ++feature)
	{
		dmMatrixView<Scalar> matrOut = outView(feature);
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
					matrOut(i, j) += sum;
				}
			}
		}
	}*/

	for (size_t feature = 0; feature < outView.Depth(); ++feature)
	{
		dmMatrixView<Scalar> matrOut = outView(feature);
		Scalar B = m_bias(0, 0, feature);
		for (size_t i = 0; i < matrOut.RowCount(); ++i)
		{
			for (size_t j = 0; j < matrOut.ColCount(); ++j)
			{
				matrOut(i, j) += B;
			}
		}
	}
}

template<typename Scalar>
void mwConv2dLayer<Scalar>::MapData(Scalar* weights, Scalar* gradient, Scalar* wokSpace)
{
	m_weights.SetView(weights);
	m_bias.SetView(weights + m_weights.Size());
	m_grads.SetView(gradient);
	m_biasGrads.SetView(gradient + m_grads.Size());
	m_workSpace.SetView(wokSpace);
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