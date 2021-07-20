#include "StdAfx.h"
#include <vector>
#include "CNN/include/mwAdamOptimizer.hpp"

template<typename Scalar>
void mwAdamOptimizer<Scalar>::Update(const std::vector<Scalar>& grads, std::vector<Scalar>& weights)
{
	if (m_v.size() != grads.size())
	{
		m_v = std::vector<Scalar>(grads.size(), Scalar(0));
		m_m = std::vector<Scalar>(grads.size(), Scalar(0));
	}
	const Scalar beta1(Scalar(0.9));
	const Scalar beta2(Scalar(0.999));
	m_b1t *= beta1;
	m_b2t *= beta2;
	for (size_t i = 0; i < weights.size(); ++i)
	{
		Scalar curLR = m_learningRate * std::sqrt(Scalar(1) - m_b2t) / (Scalar(1) - m_b1t);
		m_m[i] = beta1 * m_m[i] + (Scalar(1) - beta1) * grads[i];
		m_v[i] = beta2 * m_v[i] + (Scalar(1) - beta2) * grads[i] * grads[i];
		weights[i] -= curLR * m_m[i] / (m_epsilon + std::sqrt(m_v[i]));
	}
}

template struct mwAdamOptimizer<double>;
template struct mwAdamOptimizer<float>;

