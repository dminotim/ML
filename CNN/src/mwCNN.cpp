#include "StdAfx.h"
#include "CNN/include/mwCNN.hpp"
#include <iostream>

template<typename Scalar>
mwCNN<Scalar>::mwCNN()
{
}

template <class Scalar>
Scalar GetAverageVal(const mwTensor<Scalar>& ten)
{
	Scalar res = 0;
	for (Scalar item : ten.m_values)
	{
		res += item;
	}
	return res / ten.m_values.size();
}

template<typename Scalar>
Scalar mwCNN<Scalar>::TrainBatch(
	std::shared_ptr<mwOptimizer<Scalar>> opt,
	std::shared_ptr<mwLossFunction<Scalar>> loss,
	const std::vector<mwTensorView<Scalar>>& x,
	const std::vector<mwTensorView<Scalar>>& y,
	const size_t startIdx,
	const size_t endIdx)
{
	ValidateFinalization();
	Scalar avError = 0;
	std::vector<Scalar> gradSum(m_weightsSpace.size(), 0);
	for (size_t i = startIdx; i < endIdx; ++i)
	{
		SetZeros();
		Forward(x[i]);
		const Scalar currentLoss = GetAverageVal(loss->CalcCost(m_layers.back()->Output(), y[i]));
		mwTensor<Scalar> delta = loss->CalcDelta(m_layers.back()->Output(), y[i]);
		mwTensorView<Scalar> deltaView = delta;
		Backprop(deltaView);
		CalcGrads(deltaView);
		for (size_t j = 0; j < m_gradsSpace.size(); ++j)
		{
			gradSum[j] += m_gradsSpace[j];
		}
		avError += currentLoss;
	}
	const size_t bSize = (endIdx - startIdx);
	for (size_t i = 0; i < gradSum.size(); ++i)
	{
		gradSum[i] /= Scalar(bSize);
	}
	opt->Update(gradSum, m_weightsSpace);
	return avError / Scalar(bSize);
}

template<typename Scalar>
void mwCNN<Scalar>::ValidateFinalization() const
{
	if (!m_isFinalized)
		throw std::exception("Network should be finalized");
}

template<typename Scalar>
bool mwCNN<Scalar>::IsLayerShouldBeSkippedInPrediction(std::shared_ptr<layers::mwLayer<Scalar>> layer) const
{
	if (layer->GetType() == layers::mwLayerType::DROP_OUT)
		return true;
	return false;
}

template<typename Scalar>
void mwCNN<Scalar>::CalcGrads(const mwTensorView<Scalar>& delta)
{
	ValidateFinalization();
	m_layers.back()->CalcGrads(delta);
	for (int i = int(m_layers.size()) - 2; i >= 0; i--)
	{
		m_layers[i]->CalcGrads(m_layers[i + 1]->GetDerivatives());
	}
}

template<typename Scalar>
void mwCNN<Scalar>::Backprop(const mwTensorView<Scalar>& delta)
{
	ValidateFinalization();
	m_layers.back()->Backprop(delta);
	for (int i = int(m_layers.size()) - 2; i >= 0; i--)
	{
		m_layers[i]->Backprop(m_layers[i + 1]->GetDerivatives());
	}
}

template<typename Scalar>
void mwCNN<Scalar>::Forward(const mwTensorView<Scalar>& x)
{
	ValidateFinalization();
	if (m_layers.empty())
		return;
	m_layers[0]->Forward(x);
	for (size_t i = 1; i < m_layers.size(); ++i)
	{
		m_layers[i]->Forward(m_layers[i - 1]->Output());
	}
}

template<typename Scalar>
void mwCNN<Scalar>::SetZeros()
{
	ValidateFinalization();
	for (size_t i = 0; i < m_layers.size(); ++i)
	{
		m_layers[i]->SetDeltaToZero();
	}
}

template<typename Scalar>
void mwCNN<Scalar>::Finalize()
{
	size_t curMapIdx = 0;
	for (auto& lptr : m_layers)
	{
		const size_t dataSize = lptr->OptimizedParamsCount();
		if (dataSize == 0)
			continue;
		lptr->MapData(
			&m_weightsSpace[curMapIdx], &m_gradsSpace[curMapIdx]);
		curMapIdx += dataSize;
	}
	m_isFinalized = true;
	for (auto& lptr : m_layers)
	{
		lptr->Init();
	}
}

template<typename Scalar>
mwTensor<Scalar> mwCNN<Scalar>::Cost(const mwTensorView<Scalar>& x,
	const mwTensorView<Scalar>& expectedY,
	std::shared_ptr<mwLossFunction<Scalar>> loss) const
{
	ValidateFinalization();
	m_layers[0]->Forward(x);
	for (size_t i = 1; i < m_layers.size(); ++i)
	{
		m_layers[i]->Forward(m_layers[i - 1]->Output());
	}
	return loss->CalcCost(m_layers.back()->Output(), expectedY);
}

template<typename Scalar>
std::vector<Scalar> mwCNN<Scalar>::GetAnaliticDeltas(
	std::shared_ptr<mwLossFunction<Scalar>> loss,
	const mwTensorView<Scalar>& x,
	const mwTensorView<Scalar>& y)
{
	ValidateFinalization();
	SetZeros();
	Forward(x);
	mwTensor<Scalar> delta = loss->CalcDelta(m_layers.back()->Output(), y);
	Backprop(delta);
	CalcGrads(delta);
	return m_gradsSpace;
}

template<typename Scalar>
Scalar mwCNN<Scalar>::GetNumericDeltas(std::shared_ptr<mwLossFunction<Scalar>> loss,
	const mwTensorView<Scalar>& x,
	const mwTensorView<Scalar>& y,
	size_t idx,
	Scalar h)
{
	ValidateFinalization();
	Scalar saved = m_weightsSpace[idx];
	m_weightsSpace[idx] = saved + h;
	Scalar costFwd = GetAverageVal(this->Cost(x, y, loss));
	m_weightsSpace[idx] = saved - h;
	Scalar costBck = GetAverageVal(this->Cost(x, y, loss));
	m_weightsSpace[idx] = saved;
	return (costFwd - costBck) / h / 2;
}

template<typename Scalar>
std::vector<Scalar> mwCNN<Scalar>::GetNumericDeltas(
	std::shared_ptr<mwLossFunction<Scalar>> loss,
	const mwTensorView<Scalar>& x,
	const mwTensorView<Scalar>& y)
{
	ValidateFinalization();
	SetZeros();
	std::vector<Scalar> gradNum(m_weightsSpace.size());
	for (size_t i = 0; i < gradNum.size(); i++) {
		gradNum[i] = this->GetNumericDeltas(loss, x, y, i, Scalar(1e-8));
	}
	return gradNum;
}

template<typename Scalar>
mwTensorView<Scalar> mwCNN<Scalar>::Predict(const mwTensorView<Scalar>& x) const
{
	ValidateFinalization();
	m_layers[0]->Forward(x);
	mwTensorView<Scalar> prevOut = m_layers[0]->Output();
	for (size_t i = 1; i < m_layers.size(); ++i)
	{
		if (IsLayerShouldBeSkippedInPrediction(m_layers[i]))
			continue;
		m_layers[i]->Forward(prevOut);
		prevOut = m_layers[i]->Output();
	}
	return m_layers.back()->Output();
}

template<typename Scalar>
void mwCNN<Scalar>::AddLayer(const std::shared_ptr<layers::mwLayer<Scalar>> layer)
{
	m_isFinalized = false;
	const size_t paramsCount = layer->OptimizedParamsCount();
	m_weightsSpace.resize(m_weightsSpace.size() + paramsCount);
	m_gradsSpace.resize(m_gradsSpace.size() + paramsCount);
	m_layers.push_back(layer);
}

template<typename Scalar>
void mwCNN<Scalar>::Fit(const std::vector<mwTensorView<Scalar>>& x,
	const std::vector<mwTensorView<Scalar>>& y,
	std::shared_ptr<mwOptimizer<Scalar>> opt,
	std::shared_ptr<mwLossFunction<Scalar>> loss,
	const size_t epochCount /*= 1*/,
	const size_t batchSize /*= 1*/)
{
	for (size_t ep = 0; ep < epochCount; ++ep)
	{
		Scalar err = 0;
		size_t steps = 0;
		for (size_t start = 0, end = x.size(), next = 0;
			start < end; start = next)
		{
			next = std::min(start + batchSize, end);
			const Scalar currentLoss = TrainBatch(opt, loss, x, y, start, next);
			err += currentLoss;
			std::cout << "epoch " << ep  << " Batch " << steps << " loss = " << currentLoss << std::endl;
			++steps;
		}
		std::cout << "epoch " << ep << " loss=" << err / Scalar(steps) << std::endl;
		/*if(ep % 1000 == 0)
		std::cout << "epoch " << ep << " loss=" << err / Scalar(steps) << std::endl;*/
	}
}

template struct mwCNN<double>;
template struct mwCNN<float>;
