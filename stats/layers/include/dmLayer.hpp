#pragma once
#include "dmOptimizer.hpp"
#include <functional>
#include <vector>

namespace dmNeural
{
class dmLayer
{
public:
	dmLayer(const size_t sizeIn, const size_t sizeOut)
		:m_inputSize(sizeIn), m_outputSize(sizeOut)
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

	virtual void Init(const std::function<double()>& rnd) = 0;
	virtual void Forward(const std::vector<double>& prevLayerWeights) = 0;
	virtual void SetOut(const std::vector<double>& /*output*/)
	{
	};
	virtual std::vector<double> Backprop(const std::vector<double>& input,
		const std::vector<double>& nextLayerGrads) = 0;
	virtual const std::vector<double>& Output() const = 0;
	virtual const std::vector<double>& GetDerivatives() const = 0;

	virtual void Update(const dmOptimizer::dmOptimizerGradientDescent& opt) = 0;
protected:
	const size_t m_inputSize;
	const size_t m_outputSize;
};

} // namespace dmNeural