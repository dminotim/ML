#pragma once
#include "mwLayer.hpp"
#include "mwTensor.hpp"

namespace layers
{

template<typename Scalar>
mwLayer<Scalar>::mwLayer(const mwTensorView<Scalar>& inputShape,
	const std::function<Scalar()>& initializer)
	: m_inputShape(inputShape), m_initializer(initializer)
{
}

template<typename Scalar>
mwLayer<Scalar>::mwLayer(const size_t rowCount,
	const size_t colCount,
	const size_t depth,
	const std::function<Scalar()>& initializer)
	: m_inputShape(nullptr, rowCount, colCount, depth), m_initializer(initializer)
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