#include <vector>
#include "mwMSELossFunction.hpp"
#include "mwTensor.hpp"

template<typename Scalar>
mwTensor<Scalar> mwMSELossFunction<Scalar>::CalcDelta(const mwTensorView<Scalar>& values, const mwTensorView<Scalar>& expected)
{
	mwTensor<Scalar> delta(values.RowCount(), values.ColCount(), values.Depth());
	mwVectorView<Scalar> valuesVec = values.ToVectorView();
	mwVectorView<Scalar> expectedVec = expected.ToVectorView();
	mwVectorView<Scalar> deltaVec = delta.ToView().ToVectorView();
	for (size_t i = 0; i < valuesVec.size(); ++i)
	{
		deltaVec[i] = 2 * (valuesVec[i] - expectedVec[i]);
	}
	return delta;
}

template <class Scalar>
Scalar Sqr(const Scalar val)
{
	return val * val;
}

template<typename Scalar>
mwTensor<Scalar>  mwMSELossFunction<Scalar>::CalcCost(const mwTensorView<Scalar>& values, const mwTensorView<Scalar>& expected)
{
	mwTensor<Scalar> res(values.RowCount(), values.ColCount(), values.Depth());
	mwVectorView<Scalar> erroVec = res.ToView().ToVectorView();
	mwVectorView<Scalar> valuesVec = values.ToVectorView();
	mwVectorView<Scalar> expectedVec = expected.ToVectorView();

	for (size_t i = 0; i < valuesVec.size(); ++i)
	{
		erroVec[i] = Sqr(valuesVec[i] - expectedVec[i]);
	}
	return res;
}

template struct mwMSELossFunction<double>;
template struct mwMSELossFunction<float>;