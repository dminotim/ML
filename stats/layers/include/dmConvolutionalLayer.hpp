#pragma once
#include "dmOptimizer.hpp"
#include "dmLayer.hpp"
#include <functional>
#include <vector>
#include <sstream>

namespace dmNeural
{

class dmConvolutionalLayer : public dmLayer
{
public:
	dmConvolutionalLayer(const size_t imageInputW,
		const size_t imageInputH, const size_t kernelSize)
		: dmLayer(imageInputW * imageInputH,
			(imageInputW - 2) * (imageInputH - 2),
			dmLayerType::CONVOLUTION),
		m_weights(nullptr, kernelSize, kernelSize),
		m_z(m_outputSize, 1),
		m_dW(m_inputSize, 1),
		m_grads(nullptr, kernelSize * kernelSize),
		m_imageColCount(imageInputW),
		m_imageRowCount(imageInputH),
		m_kernelSize(kernelSize)
	{
	}

	void Forward(const std::vector<double>& input)
	{
		const size_t step = m_kernelSize - 1;
		size_t idx = 0;
		for (size_t i = 0; i + step < m_imageRowCount; ++i)
		{
			for (size_t j = 0; j + step < m_imageColCount; ++j)
			{
				m_z[idx] = 0;
				for (size_t ki = 0; ki < m_weights.GetRowCount(); ++ki)
				{
					for (size_t kj = 0; kj < m_weights.GetColCount(); ++kj)
					{
						m_z[idx] +=
							input[(i + ki) * m_imageColCount + (j + kj)] * m_weights(ki, kj);
					}
				}
				++idx;
			}
		}
	}

	void Backprop(const std::vector<double>& input,
		const std::vector<double>& nextDeriv)
	{
		const size_t step = m_kernelSize - 1;
		size_t idx = 0;
		m_dW.assign(m_dW.size(), 0.);
		for (size_t i = 0; i + step < m_imageRowCount; ++i)
		{
			for (size_t j = 0; j + step < m_imageColCount; ++j)
			{
				for (size_t ki = 0; ki < m_weights.GetRowCount(); ++ki)
				{
					for (size_t kj = 0; kj < m_weights.GetColCount(); ++kj)
					{
						const size_t dwIdx = (i + ki) * m_imageColCount + (j + kj);
						m_dW[dwIdx] += m_weights(ki, kj) * nextDeriv[idx];
					}
				}
				++idx;
			}
		}
	}

	void CalcGrads(const std::vector<double>& input,
		const std::vector<double>& nextDeriv)
	{
		for (size_t i = 0; i < m_grads.size(); ++i)
		{
			m_grads[i] = 0;
		}
		const size_t step = m_kernelSize - 1;
		size_t idx = 0;
		for (size_t i = 0; i + step < m_imageRowCount; ++i)
		{
			for (size_t j = 0; j + step < m_imageColCount; ++j)
			{
				size_t gradIdx = 0;
				for (size_t ki = 0; ki < m_weights.GetRowCount(); ++ki)
				{
					for (size_t kj = 0; kj < m_weights.GetColCount(); ++kj)
					{
						m_grads[gradIdx++] +=
							input[(i + ki) * m_imageColCount + (j + kj)] * nextDeriv[idx];
					}
				}
				++idx;
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

	const size_t GetKernelSize() const {
		return m_kernelSize;
	}

	const size_t GetImageW() const {
		return m_imageColCount;
	}

	const size_t GetImageH() const {
		return m_imageRowCount;
	}

	dmLayerType GetType() const
	{
		return this->m_type;
	}

	size_t DataCapasity() const
	{
		return m_kernelSize * m_kernelSize;
	}

	void MapData(double* wSpace, double* gSpace, const size_t size)
	{
		m_weights.set_view(wSpace, m_kernelSize, m_kernelSize);
		m_grads.set_view(gSpace, size);
	}

private:
	dmBlock<double> m_weights;
	dmVectorView<double> m_grads;
	std::vector<double> m_dW;
	std::vector<double> m_z;
	const size_t m_imageColCount;
	const size_t m_imageRowCount;
	const size_t m_kernelSize;
};

} // namespace dmNeural