#pragma once
#include "dmOptimizer.hpp"
#include "dmLayer.hpp"
#include <functional>
#include <vector>

namespace dmNeural
{

class dmFullyConnectedLayer : public dmLayer
{
public:
	dmFullyConnectedLayer(const size_t sizeIn, const size_t sizeOut)
		: dmLayer(sizeIn, sizeOut),
		m_weights(m_outputSize, m_inputSize),
		m_derivatives(m_outputSize* m_inputSize)
	{
	}
	void Init(const std::function<double()>& rnd)
	{
		m_weights = dmMatrix<double>(m_outputSize, m_inputSize);
		m_dW = std::vector<double>(m_inputSize, 0);
		m_derivatives = std::vector<dmOptimizer::dmGrad>(m_outputSize * m_inputSize);
		for (double& v : m_weights.m_values)
		{
			v = rnd();
		}
	}

	void Forward(const std::vector<double>& input)
	{
		m_z = (m_weights * input);
	}

	std::vector<double> Backprop(const std::vector<double>& input,
		const std::vector<double>& nextLayerGrads)
	{
		m_dW.assign(m_dW.size(), 0.);
		for (size_t i = 0; i < m_inputSize; ++i)
		{
			for (size_t j = 0; j < m_outputSize; ++j)
			{
				m_derivatives[j * m_inputSize + i].m_grad = input[i] * nextLayerGrads[j];
				m_dW[i] += m_weights(j, i) * nextLayerGrads[j];
			}
		}
		return m_dW;
	}

	void Update(const dmOptimizer::dmOptimizerGradientDescent& opt)
	{
		opt.Update(m_derivatives, m_weights.m_values);
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
	dmMatrix<double> m_weights;
	std::vector<double> m_dW;
	std::vector<dmOptimizer::dmGrad> m_derivatives;
	std::vector<double> m_z;
};

} // namespace dmNeural