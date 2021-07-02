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
		: dmLayer(size, size, dmLayerType::OUTPUT), m_expected(expected), m_dW(size), m_z(size)
	{
	}

	void SetOut(const std::vector<double>& out)
	{
		m_expected = out;
	}

	void Forward(const std::vector<double>& input)
	{
		for (size_t i = 0; i < m_z.size(); ++i)
		{
			m_z[i] = (input[i] - m_expected[i]) * (input[i] - m_expected[i]);
		}
	}

	void Backprop(const std::vector<double>& input,
		const std::vector<double>& /*nextLayerGrads*/)
	{
		for (size_t i = 0; i < input.size(); ++i)
		{
			m_dW[i] = 2 * (input[i] - m_expected[i]);
		}
	}

	const std::vector<double>& Output() const
	{
		return m_z;
	}

	const std::vector<double>& GetDerivatives() const
	{
		return m_dW;
	}

	void CalcGrads(const std::vector<double>& /*input*/,
		const std::vector<double>& /*nextDeriv*/)
	{
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
	std::vector<double> m_expected;
	std::vector<double> m_dW;
	std::vector<double> m_z;
};

} // namespace dmNeural