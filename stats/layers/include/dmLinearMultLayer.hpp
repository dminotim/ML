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
		: dmLayer(size, size, dmLayerType::LINEAR_MULT),
		m_weights(nullptr, size),
		m_z(size),
		m_dW(size),
		m_grads(nullptr, size)
	{
	}

	void Forward(const std::vector<double>& input)
	{
		for (size_t i = 0; i < input.size(); ++i)
		{
			m_z[i] = input[i] * m_weights[i];
		}
	}

	void Backprop(const std::vector<double>& input,
		const std::vector<double>& nextLayerGrads)
	{
		for (size_t i = 0; i < nextLayerGrads.size(); ++i)
		{
			m_dW[i] = nextLayerGrads[i] * m_weights[i];
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

	void CalcGrads(const std::vector<double>& input,
		const std::vector<double>& nextDeriv)
	{
		for (size_t i = 0; i < nextDeriv.size(); ++i)
		{
			m_grads[i] = input[i] * nextDeriv[i];
		}
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