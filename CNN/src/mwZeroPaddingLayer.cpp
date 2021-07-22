#include "mwZeroPaddingLayer.hpp"

namespace layers
{

template<typename Scalar>
mwZeroPaddingLayer<Scalar>::mwZeroPaddingLayer(
	const size_t paddingSize,
	const mwTensorView<Scalar>& inputShape)
	: mwLayer(inputShape),
	m_padding(paddingSize),
	m_delta(inputShape.RowCount(), inputShape.ColCount(), inputShape.Depth()),
	m_out(inputShape.RowCount() + paddingSize * 2,
		inputShape.ColCount() + paddingSize * 2, inputShape.Depth())
{
}

template<typename Scalar>
void mwZeroPaddingLayer<Scalar>::Init()
{
}

template<typename Scalar>
void mwZeroPaddingLayer<Scalar>::SetDeltaToZero()
{
	m_delta.ToView().SetToZero();
}

template<typename Scalar>
const mwTensorView<Scalar> mwZeroPaddingLayer<Scalar>::Input() const
{
	return m_in;
}

template<typename Scalar>
const mwTensorView<Scalar> mwZeroPaddingLayer<Scalar>::GetDerivatives() const
{
	const mwTensorView<Scalar> tensor = m_delta;
	return tensor;
}

template<typename Scalar>
const mwTensorView<Scalar> mwZeroPaddingLayer<Scalar>::Output() const
{
	const mwTensorView<Scalar> tensor = m_out;
	return tensor;
}

template<typename Scalar>
void mwZeroPaddingLayer<Scalar>::Backprop(const mwTensorView<Scalar>& nextDelta)
{
	mwTensorView<Scalar> outView = m_out.ToView();
	mwTensorView<Scalar> deltaView = m_delta.ToView();
	for (size_t feature = 0; feature < outView.Depth(); ++feature)
	{
		dmMatrixView<Scalar> nextDeltaMatr = nextDelta(feature);
		dmMatrixView<Scalar> matrIn = m_in(feature);
		dmMatrixView<Scalar> matrDelta = deltaView(feature);
		for (size_t i = 0; i < matrIn.RowCount(); ++i)
		{
			for (size_t j = 0; j < matrIn.ColCount(); ++j)
			{
				matrDelta(i, j) += nextDeltaMatr(i + m_padding, j + m_padding);
			}
		}
	}
}

template<typename Scalar>
void mwZeroPaddingLayer<Scalar>::CalcGrads(const mwTensorView<Scalar>& /*nextDelta*/)
{
}

template<typename Scalar>
void mwZeroPaddingLayer<Scalar>::Forward(const mwTensorView<Scalar>& input)
{
	m_in = input;
	mwTensorView<Scalar> outView = m_out.ToView();
	outView.SetToZero();

	for (size_t feature = 0; feature < input.Depth(); ++feature)
	{
		dmMatrixView<Scalar> matrOut = outView(feature);
		dmMatrixView<Scalar> matrIn = input(feature);
		for (size_t i = 0; i < matrIn.RowCount(); ++i)
		{
			for (size_t j = 0; j < matrIn.ColCount(); ++j)
			{
				matrOut(i + m_padding, j + m_padding) = matrIn(i, j);
			}
		}
	}
}

template<typename Scalar>
void mwZeroPaddingLayer<Scalar>::MapData(Scalar* /*weights*/, Scalar* /*gradient*/, Scalar* /*wokSpace*/)
{
}

template<typename Scalar>
size_t mwZeroPaddingLayer<Scalar>::OptimizedParamsCount() const
{
	return 0;
}

template<typename Scalar>
mwLayerType mwZeroPaddingLayer<Scalar>::GetType() const
{
	return mwLayerType::ZERO_PADDING;
}

template<typename Scalar>
mwTensorView<Scalar> mwZeroPaddingLayer<Scalar>::GetOutShape() const
{
	const mwTensorView<Scalar>& tensor = m_out;
	return tensor;
}

template<typename Scalar>
size_t mwZeroPaddingLayer<Scalar>::Padding() const { return m_padding; }

template struct mwZeroPaddingLayer<double>;
template struct mwZeroPaddingLayer<float>;

}