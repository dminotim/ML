#pragma  once
#include "mwTensor.hpp"
#include "mwInitialization.hpp"

template<typename Scalar>
struct mwHeInitialization: public mwInitialization<Scalar>
{
	mwHeInitialization();
	void Init(
		const size_t inputCount,
		mwTensorView<Scalar>& toInit) override;
};

