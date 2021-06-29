#pragma once
#include <vector>
#include "dmMatrix.hpp"

namespace dmOptimizer
{

	struct dmGrad
	{
		dmGrad()
			:m_oldGrad(0), m_grad(0)
		{
		}
		double m_oldGrad;
		double m_grad;
	};

	struct dmOptimizerGradientDescent
	{
		dmOptimizerGradientDescent(const double learningRate,
			const double momentum,
			const double weightDecay)
			: m_learningRate(learningRate),
			m_momentum(momentum),
			m_weightDecay(weightDecay)
		{
		}

		void Update(std::vector<dmGrad>& grads, std::vector<double>& weights) const
		{
			for(size_t i = 0; i < weights.size(); ++i)
			{
				//const double m = (grads[i].m_grad + grads[i].m_oldGrad * m_momentum);
				weights[i] -= m_learningRate * (grads[i].m_grad);
					//m_learningRate * m_weightDecay * weights[i];
	/*			weights[i] -= m_learningRate* m +
					m_learningRate * m_weightDecay * weights[i];
				grads[i].m_oldGrad = (grads[i].m_grad + grads[i].m_oldGrad * m_momentum);*/
			}
		}

		double m_learningRate;
		double m_momentum;
		double m_weightDecay;
	};
}