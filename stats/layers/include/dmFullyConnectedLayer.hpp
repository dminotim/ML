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
		: dmLayer(sizeIn, sizeOut, dmLayerType::FULLY_CONNECTED),
		m_weights(nullptr, m_outputSize, m_inputSize),
		m_grads(nullptr, m_outputSize* m_inputSize),
		m_dW(m_inputSize),
		m_z(m_outputSize)
	{
	}

	void Forward(const std::vector<double>& input)
	{
		m_weights.MultiplyInPlace(input, m_z);
	}

	void Backprop(const std::vector<double>& input,
		const std::vector<double>& nextLayerGrads)
	{
		for (size_t i = 0; i < m_inputSize; ++i)
		{
			m_dW[i] = 0;
			for (size_t j = 0; j < m_outputSize; ++j)
			{
				m_dW[i] += m_weights(j, i) * nextLayerGrads[j];
			}
		}
	}

	void CalcGrads(const std::vector<double>& input,
		const std::vector<double>& nextDeriv)
	{
		for (size_t i = 0; i < m_inputSize; ++i)
		{
			for (size_t j = 0; j < m_outputSize; ++j)
			{
				m_grads[j * m_inputSize + i] = input[i] * nextDeriv[j];
			}
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
		return m_outputSize * m_inputSize;
	}

	void MapData(double* wSpace, double* gSpace, const size_t size)
	{
		m_weights.set_view(wSpace, m_outputSize, m_inputSize);
		m_grads.set_view(gSpace, size);
	}
	
private:
	dmBlock<double> m_weights;
	dmVectorView<double> m_grads;
	std::vector<double> m_z;
	std::vector<double> m_dW;
};

} // namespace dmNeural