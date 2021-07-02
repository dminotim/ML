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
		: dmLayer(size, size, dmLayerType::RE_LU), m_z(size, 1), m_dW(size, 1)
	{
	}
	void Forward(const std::vector<double>& input)
	{
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

	void Backprop(const std::vector<double>& input,
		const std::vector<double>& nextLayerGrads)
	{
		for (size_t i = 0; i < nextLayerGrads.size(); ++i)
		{
			m_dW[i] = (input[i] < 0) ? (0) : (1. * nextLayerGrads[i]);
		}
	}

	const std::vector<double>& GetDerivatives() const
	{
		return m_dW;
	}
	void CalcGrads(const std::vector<double>& /*input*/,
		const std::vector<double>& /*nextDeriv*/)
	{
	}

	const std::vector<double>& Output() const
	{
		return m_z;
	}

	dmLayerType GetType() const
	{
		return this->m_type;
	}

	size_t DataCapasity() const
	{
		return 0;
	}

	void MapData(double* /*wSpace*/, double* /*gSpace*/, const size_t /*size*/)
	{
	}
private:
	std::vector<double> m_dW;
	std::vector<double> m_z;
};

} // namespace dmNeural