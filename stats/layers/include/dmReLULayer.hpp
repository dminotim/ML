#pragma once
#include "dmOptimizer.hpp"
#include "dmLayer.hpp"
#include <functional>
#include <vector>

namespace dmNeural
{
class dmReLULayer : public dmLayer
{
public:
	dmReLULayer(const size_t size)
		: dmLayer(size, size), m_z(size, 1), m_dW(size, 1)
	{
	}

	void Init(const std::function<double()>& /*rnd*/)
	{
	}

	void Forward(const std::vector<double>& input)
	{
		m_z = std::vector<double>(input.size());
		for (size_t i = 0; i < input.size(); ++i)
		{
			double v = input[i];
			if (v < 0)
			{
				v = 0;
			}
			m_z[i] = v;
		}
	}

	std::vector<double> Backprop(const std::vector<double>& input,
		const std::vector<double>& nextLayerGrads)
	{
		for (size_t i = 0; i < nextLayerGrads.size(); ++i)
		{
			m_dW[i] = (input[i] < 0) ? (0) : (1. * nextLayerGrads[i]);
		}
		return m_dW;
	}

	void Update(const dmOptimizer::dmOptimizerGradientDescent& /*opt*/)
	{
		//opt.Update(m_dW, m_weights);
	}

	const std::vector<double>& GetDerivatives() const
	{
		return m_dW;
	}

	const std::vector<double>& Output() const
	{
		return m_z;
	}
private:
	std::vector<double> m_dW;
	std::vector<double> m_z;

};

} // namespace dmNeural