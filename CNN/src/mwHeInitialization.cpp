#include "mwHeInitialization.hpp"
#include <complex>
#include <random>
#include "mwTensor.hpp"


template<typename Scalar>
void mwHeInitialization<Scalar>::Init(const size_t inputCount, mwTensorView<Scalar>& toInit)
{
	const Scalar mean = 0.0;
	const Scalar deviation = std::sqrt(
		Scalar(2) / static_cast<Scalar>(inputCount));
	std::normal_distribution<Scalar> randNormal(mean, deviation);

	mwVectorView<Scalar> inVec = toInit.ToVectorView();
	for (size_t i = 0; i < inVec.size(); ++i)
	{
		inVec[i] = randNormal(m_engine);
	}
}

template<typename Scalar>
mwHeInitialization<Scalar>::mwHeInitialization()
{
}

template struct mwHeInitialization<float>;
template struct mwHeInitialization<double>;
