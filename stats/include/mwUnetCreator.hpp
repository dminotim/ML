#pragma once
#include <vector>
#include "mwCNN.hpp"
#include "mwTensor.hpp"

namespace mwUnetCreator
{
template<class Scalar>
void Create(const mwTensorView<Scalar>& inputShape, mwCNN<Scalar>& model);
};
