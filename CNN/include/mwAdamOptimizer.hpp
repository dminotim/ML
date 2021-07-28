#pragma  once
#include "mwOptimizer.hpp"
#include <vector>

template<typename Scalar>
struct mwAdamOptimizer : public mwOptimizer<Scalar>
{
	mwAdamOptimizer(const Scalar lr = Scalar(1e-3), const Scalar epsilon = Scalar(1e-7))
		: m_learningRate(lr),
		m_m(),
		m_v(),
		m_b1t(Scalar(0.9)),
		m_b2t(Scalar(0.999)),
		m_epsilon(epsilon)
	{
	}

	void Update(const std::vector<Scalar>& grads, std::vector<Scalar>& weights) override;
	std::vector<Scalar> m_v;
	std::vector<Scalar> m_m;
	Scalar m_b1t;
	Scalar m_b2t;
	Scalar m_learningRate;
	Scalar m_epsilon;
};
