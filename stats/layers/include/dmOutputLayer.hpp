#pragma once
#include "dmOptimizer.hpp"
#include "dmLayer.hpp"
#include <functional>
#include <vector>

namespace dmNeural
{

class dmOutputLayer : public dmLayer
{
public:
	dmOutputLayer(const size_t size, const std::vector<double>& expected)
		: dmLayer(size, size), m_expected(expected), m_dW(size), m_z(size)
	{
	}

	void Init(const std::function<double()>& /*rnd*/)
	{
	}
	void SetOut(const std::vector<double>& out)
	{
		m_expected = out;
	}

	void Forward(const std::vector<double>& prevLayerWeights)
	{
		m_z = prevLayerWeights;
	}

	std::vector<double> Backprop(const std::vector<double>& input,
		const std::vector<double>& /*nextLayerGrads*/)
	{
		for (size_t i = 0; i < input.size(); ++i)
		{
			m_dW[i] = 2 * (input[i] - m_expected[i]);
		}
		return m_dW;
	}

	void Update(const dmOptimizer::dmOptimizerGradientDescent& /*opt*/)
	{
		//opt.Update(m_dW, m_weights.m_values);
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
	std::vector<double> m_expected;
	std::vector<double> m_dW;
	std::vector<double> m_z;
};

} // namespace dmNeural