#pragma once
#include "dmOptimizer.hpp"
#include "dmLayer.hpp"
#include <functional>
#include <vector>
#include <sstream>

namespace dmNeural
{

class dmBiasLayer : public dmLayer
{
public:
	dmBiasLayer(const size_t size)
		: dmLayer(size, size, dmLayerType::BIAS),
		m_weights(nullptr, size),
		m_z(size, 1),
		m_dW(size, 1),
		m_grads(nullptr, size)
	{
	}

	void Forward(const std::vector<double>& input)
	{
		for (size_t i = 0; i < input.size(); ++i)
		{
			m_z[i] = input[i] + m_weights[i];
		}
	}

	void Backprop(const std::vector<double>& /*input*/,
		const std::vector<double>& nextLayerGrads)
	{
		m_dW = nextLayerGrads;
	}

	void CalcGrads(const std::vector<double>& input,
		const std::vector<double>& nextDeriv)
	{
		for (size_t i = 0; i < m_dW.size(); ++i)
		{
			m_grads[i] = m_dW[i];
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

	dmLayerType GetType() const
	{
		return this->m_type;
	}

	size_t DataCapasity() const
	{
		return m_outputSize;
	}

	void MapData(double* wSpace, double* gSpace, const size_t size)
	{
		m_weights.set_view(wSpace, size);
		m_grads.set_view(gSpace, size);
	}

private:
	dmVectorView<double> m_weights;
	dmVectorView<double> m_grads;
	std::vector<double> m_dW;
	std::vector<double> m_z;
};

} // namespace dmNeural