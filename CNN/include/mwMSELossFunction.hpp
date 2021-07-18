#pragma  once
#include "mwTensor.hpp"
#include "mwLossFunction.hpp"

template<typename Scalar>
struct mwMSELossFunction: public mwLossFunction<Scalar>
{
	mwTensor<Scalar>  CalcCost(const mwTensorView<Scalar>& values, const mwTensorView<Scalar>& expected) override;
	mwTensor<Scalar> CalcDelta(const mwTensorView<Scalar>& values, const mwTensorView<Scalar>& expected) override;
};
