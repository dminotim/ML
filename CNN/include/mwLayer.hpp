#pragma once
#include "mwTensor.hpp"
#include <functional>
#include <random>

namespace layers
{
enum class mwLayerType
{
	FULLY_CONNECTED,
	RELU,
	CONVOLUTION,
	MAX_POOL,
	UPSAMPLING,
	ZERO_PADDING,
	SIGMOID,
	DROP_OUT,
	CONCAT,
	SOFTMAX,
	UNKNOWN
};




template<typename Scalar>
struct mwLayer
{
	mwLayer(const mwTensorView<Scalar>& inputShape);
	mwLayer(const size_t rowCount,
		const size_t colCount,
		const size_t depth);
	mwTensorView<Scalar> InputShape() const;
	void InputShape(const mwTensorView<Scalar>& val);

	virtual mwTensorView<Scalar> GetOutShape() const = 0;

	virtual mwLayerType GetType() const = 0;
	virtual size_t OptimizedParamsCount() const = 0;
	virtual void Init() = 0;
	virtual void MapData(Scalar* weights, Scalar* gradient) = 0;

	virtual void Forward(const mwTensorView<Scalar>& input) = 0;
	virtual void CalcGrads(const mwTensorView<Scalar>& nextDelta) = 0;
	virtual void Backprop(const mwTensorView<Scalar>& nextDelta) = 0;
	virtual const mwTensorView<Scalar> Output() const = 0;
	virtual const mwTensorView<Scalar> Input() const = 0;
	virtual void SetDeltaToZero() = 0;
	virtual const mwTensorView<Scalar> GetDerivatives() const = 0;
protected:
	mwTensorView<Scalar> m_inputShape;
};

}