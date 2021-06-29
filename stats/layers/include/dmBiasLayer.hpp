#pragma once
#include "dmOptimizer.hpp"
#include "dmLayer.hpp"
#include <functional>
#include <vector>

namespace dmNeural
{

class dmBiasLayer : public dmLayer
{
public:
	dmBiasLayer(const size_t size)
		: dmLayer(size, size),
		m_weights(size, 1),
		m_z(size, 1),
		m_dW(size, 1),
		m_derivatives(size)
	{
	}
	void Init(const std::function<double()>& rnd)
	{
		m_weights = std::vector<double>(m_outputSize, 1);
		for (double& v : m_weights)
		{
			v = rnd();
		}
	}

	void Forward(const std::vector<double>& input)
	{
		// z = in + b
		for (size_t i = 0; i < input.size(); ++i)
		{
			m_z[i] = input[i] + m_weights[i];
		}
	}

	std::vector<double> Backprop(const std::vector<double>& /*input*/,
		const std::vector<double>& nextLayerGrads)
	{
		m_dW = nextLayerGrads;
		for (size_t i = 0; i < m_dW.size(); ++i)
		{
			m_derivatives[i].m_grad = m_dW[i];
		}
		return m_dW;
	}

	void Update(const dmOptimizer::dmOptimizerGradientDescent& opt)
	{
		opt.Update(m_derivatives, m_weights);
	}

	const std::vector<double>& Output() const
	{
		return m_z;
	}

	const std::vector<double>& GetDerivatives() const
	{
		return m_dW;
	}
private:
	std::vector<double> m_weights;
	std::vector<double> m_dW;
	std::vector <dmOptimizer::dmGrad> m_derivatives;
	std::vector<double> m_z;

};

} // namespace dmNeural