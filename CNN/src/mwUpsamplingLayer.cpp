#include "mwUpsamplingLayer.hpp"

namespace layers
{

template<typename Scalar>
mwUpsamplingLayer<Scalar>::mwUpsamplingLayer(const size_t kernel, const mwTensorView<Scalar>& inputShape)
	:mwLayer(inputShape),
	m_out(inputShape.RowCount() * 2, inputShape.ColCount() * 2, inputShape.Depth()),
	m_delta(inputShape.RowCount(), inputShape.ColCount(), inputShape.Depth()),
	m_kernel(kernel)
{
}

template<typename Scalar>
size_t mwUpsamplingLayer<Scalar>::Kernel() const
{
	return m_kernel;
}

template<typename Scalar>
void mwUpsamplingLayer<Scalar>::Init()
{
}

template<typename Scalar>
const mwTensorView<Scalar> mwUpsamplingLayer<Scalar>::GetDerivatives() const
{
	const mwTensorView<Scalar>& tensor = m_delta;
	return tensor;
}

template<typename Scalar>
const mwTensorView<Scalar> mwUpsamplingLayer<Scalar>::Output() const
{
	const mwTensorView<Scalar>& tensor = m_out;
	return tensor;
}

template<typename Scalar>
void mwUpsamplingLayer<Scalar>::SetDeltaToZero()
{
	m_delta.ToView().SetToZero();
}

template<typename Scalar>
const mwTensorView<Scalar> mwUpsamplingLayer<Scalar>::Input() const
{
	return m_in;
}

template<typename Scalar>
void mwUpsamplingLayer<Scalar>::Backprop(const mwTensorView<Scalar>& nextDelta)
{
	for (size_t d = 0; d < m_delta.ToView().Depth(); ++d)
	{
		dmMatrixView<Scalar> matrDelta = m_delta.ToView()(d);
		dmMatrixView<Scalar> matrOutDelta = nextDelta(d);
		for (size_t i = 0; i < matrOutDelta.RowCount(); ++i)
		{
			for (size_t j = 0; j < matrOutDelta.ColCount(); ++j)
			{
				matrDelta(i / m_kernel, j / m_kernel) += matrOutDelta(i, j);
			}
		}
	}
}

template<typename Scalar>
void mwUpsamplingLayer<Scalar>::CalcGrads(const mwTensorView<Scalar>& /*nextDelta*/)
{

}

template<typename Scalar>
void mwUpsamplingLayer<Scalar>::Forward(const mwTensorView<Scalar>& input)
{
	m_in = input;
	m_out.ToView().SetToZero();
	for (size_t d = 0; d < input.Depth(); ++d)
	{
		dmMatrixView<Scalar> matrIn = input(d);
		dmMatrixView<Scalar> matrOut = m_out.ToView()(d);
		for (size_t i = 0; i < matrOut.RowCount(); ++i)
		{
			for (size_t j = 0; j < matrOut.ColCount(); ++j)
			{
				matrOut(i, j) = matrIn(i / m_kernel, j / m_kernel);
			}
		}
	}
}

template<typename Scalar>
void mwUpsamplingLayer<Scalar>::MapData(Scalar* /*weights*/, Scalar* /*gradient*/)
{
}

template<typename Scalar>
size_t mwUpsamplingLayer<Scalar>::OptimizedParamsCount() const
{
	return 0;
}

template<typename Scalar>
mwLayerType mwUpsamplingLayer<Scalar>::GetType() const
{
	return mwLayerType::UPSAMPLING;
}

template<typename Scalar>
mwTensorView<Scalar> mwUpsamplingLayer<Scalar>::GetOutShape() const
{
	const mwTensorView<Scalar>& tensor = m_out;
	return tensor;
}

template struct mwUpsamplingLayer<float>;
template struct mwUpsamplingLayer<double>;
}
