#pragma  once
#include "mwTensor.hpp"
#include "mwLayer.hpp"

namespace mwCNNUtils
{
	template <class Scalar>
	void ToColumnImage(const mwTensorView<Scalar>& src,
		const size_t kernel,
		const size_t padding,
		mwTensorView<Scalar>& dst);

	template <class Scalar>
	void FromColumnImage(const mwTensorView<Scalar>& srcCol,
		const size_t kernel,
		const size_t padding,
		mwTensorView<Scalar>& dst);

	template <class Scalar>
	void ToRowImage(const mwTensorView<Scalar>& src,
		const size_t kernel,
		const size_t padding,
		mwTensorView<Scalar>& dst);

	template <class Scalar>
	Scalar MovingAverage(Scalar avg, const size_t acc_number, Scalar value);


}

