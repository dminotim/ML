#pragma once
#include <vector>
#include "dmMatrix.hpp"

namespace dmOptimizer
{

	struct dmGrad
	{
		dmGrad()
			:m_oldGrad(0), m_grad(0), m_m(0), m_v(0)
		{
		}
		double m_oldGrad;
		double m_grad;
		double m_m;
		double m_v;
		double b1t = 0.9;
		double b2t = 0.999;
	};

	struct dmOptimizerGradientDescent
	{
		dmOptimizerGradientDescent(const double learningRate,
			const double momentum,
			const double weightDecay, const size_t size)
			: m_learningRate(learningRate),
			m_momentum(momentum),
			m_weightDecay(weightDecay),
			m_m(size, 0),
			m_v(size, 0)
		{
		}

		void Update(const std::vector<double>& grads, std::vector<double>& weights) const
		{
			
			double LR = 1e-3;
			const double beta1 = 0.9;
			const double beta2 = 0.999;
			m_b1t *= beta1;
			m_b2t *= beta2;
			for(size_t i = 0; i < weights.size(); ++i)
			{
				double curLR = LR * std::sqrt(1. - m_b2t) / (1. - m_b1t);
				m_m[i] = beta1 * m_m[i] + (1. - beta1) * grads[i];
				m_v[i] = beta2 * m_v[i] + (1. - beta2) * grads[i] * grads[i];
				//const double m = (grads[i].m_grad + grads[i].m_oldGrad * m_momentum);
				weights[i] -= curLR * m_m[i] / (1e-8 + std::sqrt(m_v[i]));
					//m_learningRate * m_weightDecay * weights[i];
				//weights[i] -= m_learningRate * grads[i];
	/*			weights[i] -= m_learningRate* m +
					m_learningRate * m_weightDecay * weights[i];
				grads[i].m_oldGrad = (grads[i].m_grad + grads[i].m_oldGrad * m_momentum);*/
			}
		}

//		void Update(std::vector<dmGrad>& grads, std::vector<double>& weights) const
//		{
//			for (size_t i = 0; i < weights.size(); ++i)
//			{
//				//const double m = (grads[i].m_grad + grads[i].m_oldGrad * m_momentum);
//				weights[i] -= m_learningRate * (grads[i].m_grad);
//				//m_learningRate * m_weightDecay * weights[i];
///*			weights[i] -= m_learningRate* m +
//				m_learningRate * m_weightDecay * weights[i];
//			grads[i].m_oldGrad = (grads[i].m_grad + grads[i].m_oldGrad * m_momentum);*/
//			}
//		}
		mutable std::vector<double> m_v;
		mutable std::vector<double> m_m;
		mutable double m_b1t = 0.9;
		mutable double m_b2t = 0.999;
		double m_learningRate;
		double m_momentum;
		double m_weightDecay;
	};
}