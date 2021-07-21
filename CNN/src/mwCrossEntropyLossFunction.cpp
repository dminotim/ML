#include <vector>
#include "mwCrossEntropyLossFunction.hpp"
#include "mwTensor.hpp"
#include <complex>
#include "mwCNNUtils.hpp"


template<typename Scalar>
mwTensor<Scalar> mwCrossEntropyLossFunction<Scalar>::CalcDelta(const mwTensorView<Scalar>& values, const mwTensorView<Scalar>& expected)
{
	mwTensor<Scalar> delta(values.RowCount(), values.ColCount(), values.Depth());
	mwVectorView<Scalar> valuesVec = values.ToVectorView();
	mwVectorView<Scalar> expectedVec = expected.ToVectorView();
	mwVectorView<Scalar> deltaVec = delta.ToView().ToVectorView();
	for (size_t j = 0; j < deltaVec.size(); ++j)
	{
		deltaVec[j] -= expectedVec[j] / (valuesVec[j]);
	}
	return delta;
}


template<typename Scalar>
mwTensor<Scalar> mwCrossEntropyLossFunction<Scalar>::CalcCost(
	const mwTensorView<Scalar>& values, const mwTensorView<Scalar>& expected)
{
	mwTensor<Scalar> res(1, 1, 1);
	mwVectorView<Scalar> erroVec = res.ToView().ToVectorView();
	mwVectorView<Scalar> valuesVec = values.ToVectorView();
	mwVectorView<Scalar> expectedVec = expected.ToVectorView();
	Scalar loss = 0;
	for (size_t i = 0; i < expectedVec.size(); i++)
	{
		const float curLoss = -expectedVec[i] * std::log(valuesVec[i]);
		loss = mwCNNUtils::MovingAverage<Scalar>(loss, i + 1, curLoss);
	}
	erroVec[0] = loss * expectedVec.size();
	return res;
}

template struct mwCrossEntropyLossFunction<double>;
template struct mwCrossEntropyLossFunction<float>;