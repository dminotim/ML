#include "CNN/include/mwConcatLayer.hpp"

namespace layers
{
template<typename Scalar>
size_t SumDepth(const std::vector< std::shared_ptr<mwLayer<Scalar>> >& layersVec)
{
	size_t sum = 0;
	for (auto layer : layersVec)
	{
		sum += layer->GetOutShape().Depth();
	}
	return sum;
}

template<typename Scalar>
mwConcatLayer<Scalar>::mwConcatLayer(
	const std::vector<std::shared_ptr<mwLayer<Scalar>>> toConcat,
	const mwTensorView<Scalar>& inputShape)
	:mwLayer(inputShape),
	m_concatenated(toConcat),
	m_out(inputShape.RowCount(), inputShape.ColCount(), SumDepth<Scalar>(toConcat) + inputShape.Depth()),
	m_delta(inputShape.RowCount(), inputShape.ColCount(), inputShape.Depth())
{
}

template<typename Scalar>
void mwConcatLayer<Scalar>::SetDeltaToZero()
{
	m_delta.ToView().SetToZero();
}

template<typename Scalar>
const mwTensorView<Scalar> mwConcatLayer<Scalar>::Input() const
{
	return m_in;
}



template<typename Scalar>
std::vector<std::shared_ptr<mwLayer<Scalar>>> mwConcatLayer<Scalar>::Concatenated() const
{
	return m_concatenated;
}

template<typename Scalar>
void mwConcatLayer<Scalar>::Init()
{
}

template<typename Scalar>
const mwTensorView<Scalar> mwConcatLayer<Scalar>::GetDerivatives() const
{
	const mwTensorView<Scalar>& tensor = m_delta;
	return tensor;
}

template<typename Scalar>
const mwTensorView<Scalar> mwConcatLayer<Scalar>::Output() const
{
	const mwTensorView<Scalar>& tensor = m_out;
	return tensor;
}

template<typename Scalar>
void mwConcatLayer<Scalar>::Backprop(const mwTensorView<Scalar>& nextDelta)
{
	size_t curD = 0;
	for (size_t lidx = 0; lidx < m_concatenated.size(); ++lidx)
	{
		mwTensorView<Scalar> subTensorDelta = nextDelta.GetSubTensorByDepth(
			curD, curD + m_concatenated[lidx]->GetOutShape().Depth());
		curD = curD + m_concatenated[lidx]->GetOutShape().Depth();
		m_concatenated[lidx]->Backprop(subTensorDelta);
	}
	mwTensorView<Scalar> inputDelta = nextDelta.GetSubTensorByDepth(
		curD, nextDelta.Depth());
	mwVectorView<Scalar> deltaVec = m_delta.ToView().ToVectorView();
	mwVectorView<Scalar> inputDeltaVec = inputDelta.ToVectorView();
	for (size_t i = 0; i < deltaVec.size(); ++i)
	{
		deltaVec[i] = inputDeltaVec[i];
	}
}

template<typename Scalar>
void mwConcatLayer<Scalar>::CalcGrads(const mwTensorView<Scalar>& nextDelta)
{
	size_t curD = 0;
	for (size_t lidx = 0; lidx < m_concatenated.size(); ++lidx)
	{
		mwTensorView<Scalar> subTensorDelta = nextDelta.GetSubTensorByDepth(
			curD, curD + m_concatenated[lidx]->GetOutShape().Depth());
		curD = curD + m_concatenated[lidx]->GetOutShape().Depth();
		m_concatenated[lidx]->CalcGrads(subTensorDelta);
	}
}

template<typename Scalar>
void mwConcatLayer<Scalar>::Forward(const mwTensorView<Scalar>& input)
{
	m_in = input;
	size_t outIdx = 0;
	mwVectorView<Scalar> outVec = m_out.ToView().ToVectorView();
	mwVectorView<Scalar> inVec = input.ToVectorView();
	for (size_t lidx = 0; lidx < m_concatenated.size(); ++lidx)
	{
		mwVectorView<Scalar> lv = m_concatenated[lidx]->Output().ToVectorView();
		for (size_t i = 0; i < lv.size(); ++i)
		{
			outVec[outIdx] = lv[i];
			++outIdx;
		}
	}
	for (size_t i = 0; i < inVec.size(); ++i)
	{
		outVec[outIdx] = inVec[i];
		++outIdx;
	}
}

template<typename Scalar>
void mwConcatLayer<Scalar>::MapData(Scalar* /*weights*/, Scalar* /*gradient*/)
{
}

template<typename Scalar>
size_t mwConcatLayer<Scalar>::OptimizedParamsCount() const
{
	return 0;
}

template<typename Scalar>
mwLayerType mwConcatLayer<Scalar>::GetType() const
{
	return mwLayerType::CONCAT;
}

template<typename Scalar>
mwTensorView<Scalar> mwConcatLayer<Scalar>::GetOutShape() const
{
	const mwTensorView<Scalar>& tensor = m_out;
	return tensor;
}

template struct mwConcatLayer<double>;
template struct mwConcatLayer<float>;

}