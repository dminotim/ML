#pragma once
#include "mwLayer.hpp"
#include "mwTensor.hpp"

namespace layers
{

template<typename Scalar>
mwLayer<Scalar>::mwLayer(const mwTensorView<Scalar>& inputShape)
	: m_inputShape(inputShape)
{
}

template<typename Scalar>
mwLayer<Scalar>::mwLayer(const size_t rowCount,
	const size_t colCount,
	const size_t depth)
	: m_inputShape(nullptr, rowCount, colCount, depth)
{
}

template<typename Scalar>
mwTensorView<Scalar> mwLayer<Scalar>::InputShape() const
{
	return m_inputShape;
}

template<typename Scalar>
void mwLayer<Scalar>::InputShape(const mwTensorView<Scalar>& val)
{
	m_inputShape = val;
}

template struct mwLayer<double>;
template struct mwLayer<float>;
}