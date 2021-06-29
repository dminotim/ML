#pragma once
#include "dmOptimizer.hpp"
#include "dmLayer.hpp"
#include <functional>
#include <vector>

namespace dmNeural
{

class dmLinearMultLayer : public dmLayer
{
public:
	dmLinearMultLayer(const size_t size, const double defaultVal)
		: dmLayer(size, size),
		m_weights(size, defaultVal),
		m_z(size),
		m_dW(size),
		m_derivatives(size)
	{
		for (double& v : m_weights)
		{
			v = defaultVal;
		}
	}

	void Init(const std::function<double()>& rnd)
	{
		m_weights = std::vector<double>(m_outputSize);
		for (double& v : m_weights)
		{
			v = rnd();
		}
	}

	void Forward(const std::vector<double>& input)
	{
		for (size_t i = 0; i < input.size(); ++i)
		{
			m_z[i] = input[i] * m_weights[i];
		}
	}

	std::vector<double> Backprop(const std::vector<double>& input,
		const std::vector<double>& nextLayerGrads)
	{
		for (size_t i = 0; i < nextLayerGrads.size(); ++i)
		{
			m_dW[i] = nextLayerGrads[i] * m_weights[i];
			m_derivatives[i].m_grad = input[i] * nextLayerGrads[i];
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
	std::vector<dmOptimizer::dmGrad> m_derivatives;
	std::vector<double> m_dW;
	std::vector<double> m_z;
};

} // namespace dmNeural