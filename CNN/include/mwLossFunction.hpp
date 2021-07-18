#pragma  once
#include "mwTensor.hpp"

template<typename Scalar>
struct mwLossFunction
{
	virtual mwTensor<Scalar> CalcCost(const mwTensorView<Scalar>& values, const mwTensorView<Scalar>& expected) = 0;
	virtual mwTensor<Scalar> CalcDelta(const mwTensorView<Scalar>& values, const mwTensorView<Scalar>& expected) = 0;
};