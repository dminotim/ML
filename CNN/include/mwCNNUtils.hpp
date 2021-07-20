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



}

