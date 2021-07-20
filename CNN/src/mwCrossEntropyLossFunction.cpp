#include <vector>
#include "mwCrossEntropyLossFunction.hpp"
#include "mwTensor.hpp"
#include <complex>
#include <iostream>

template <class Scalar>
Scalar MovingAverage(Scalar avg, const size_t acc_number, Scalar value)
{
	avg -= avg / acc_number;
	avg += value / acc_number;
	return avg;
}

template<typename Scalar>
mwTensor<Scalar> mwCrossEntropyLossFunction<Scalar>::CalcDelta(const mwTensorView<Scalar>& values, const mwTensorView<Scalar>& expected)
{
	mwTensor<Scalar> delta(values.RowCount(), values.ColCount(), values.Depth());
	mwVectorView<Scalar> valuesVec = values.ToVectorView();
	mwVectorView<Scalar> expectedVec = expected.ToVectorView();
	mwVectorView<Scalar> deltaVec = delta.ToView().ToVectorView();


	for (size_t j = 0; j < deltaVec.size(); ++j)
	{
		Scalar target = expectedVec[j];
		Scalar input = valuesVec[j];
		deltaVec[j] = -(target * Scalar(1) / (input)
			+(Scalar(1) - target) * Scalar(1) / (Scalar(1) - input) * Scalar(-1));
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
		Scalar target = expectedVec[i];
		Scalar input = valuesVec[i];
		const Scalar curLoss =
			-(target * std::log(input) + (Scalar(1) - target) * std::log(Scalar(1) - input));
		loss = MovingAverage<Scalar>(loss, i + 1, curLoss);
	}
	erroVec[0] = loss * expectedVec.size();
	return res;
}

template struct mwCrossEntropyLossFunction<double>;
template struct mwCrossEntropyLossFunction<float>;