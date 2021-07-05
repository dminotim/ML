#pragma once
#include "dmOptimizer.hpp"
#include <functional>
#include <vector>

namespace dmNeural
{

enum class dmLayerType
{
	FULLY_CONNECTED,
	RE_LU,
	BIAS,
	LINEAR_MULT,
	OUTPUT,
	TANH,
	CONVOLUTION,
	UNKNOWN
};

class dmLayer
{
public:
	dmLayer(const size_t sizeIn, const size_t sizeOut, const dmLayerType& type)
		:m_inputSize(sizeIn), m_outputSize(sizeOut), m_type(type)
	{
	}
	size_t GetInSize() const
	{
		return m_inputSize;
	}

	size_t GetOutSize() const
	{
		return m_outputSize;
	}

	virtual dmLayerType GetType() const = 0;
	virtual size_t DataCapasity() const = 0;
	virtual void MapData(double* wSpace, double* gSpace, const size_t size) = 0;
	virtual void Forward(const std::vector<double>& input) = 0;
	virtual void CalcGrads(const std::vector<double>& input,
		const std::vector<double>& nextDeriv) = 0;
	virtual void SetOut(const std::vector<double>& /*output*/)
	{
	};
	virtual void Backprop(const std::vector<double>& input,
		const std::vector<double>& nextLayerGrads) = 0;
	virtual const std::vector<double>& Output() const = 0;
	virtual const std::vector<double>& GetDerivatives() const = 0;
protected:
	size_t m_inputSize;
	size_t m_outputSize;
	const dmLayerType m_type;
};

} // namespace dmNeural