#pragma  once
#include "mwTensor.hpp"
#include <random>

template<typename Scalar>
struct mwInitialization
{
	mwInitialization()
		: m_engine(1)
	{
	}
	virtual void Init(
		const size_t inputCount,
		mwTensorView<Scalar>& toInit) = 0;
protected:
	std::default_random_engine m_engine;
};
