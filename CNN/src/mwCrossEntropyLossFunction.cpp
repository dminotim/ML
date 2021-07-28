#include <vector>
#include "mwCrossEntropyLossFunction.hpp"
#include "mwTensor.hpp"
#include <complex>
#include "mwCNNUtils.hpp"

template<typename Scalar>
Scalar Clamp(const Scalar v, const Scalar a = Scalar(1e-7), const Scalar b = Scalar(1. - 1e-7))
{
	return std::clamp(v, a, b);
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
		const Scalar predicted = Clamp(valuesVec[j]);
		const Scalar trueY = Clamp(expectedVec[j]);
		deltaVec[j] += (predicted - trueY) / (predicted * ( 1 - predicted));
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
		const Scalar predicted = Clamp(valuesVec[i]);
		const Scalar trueY = Clamp(expectedVec[i]);
		const Scalar curLoss = -(trueY * std::log(predicted)
			+ (Scalar(1) - trueY) * std::log(Scalar(1) - predicted));
		loss += curLoss;
	}
	erroVec[0] = loss;
	return res;
}

template struct mwCrossEntropyLossFunction<double>;
template struct mwCrossEntropyLossFunction<float>;