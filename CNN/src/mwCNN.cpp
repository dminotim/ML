#include "StdAfx.h"
#include "CNN/include/mwCNN.hpp"
#include <iostream>
#include <fstream>
#include "dmBinaryStream.hpp"
#include "mwLayerSerializer.hpp"
#include "dmDataReader.hpp"
#include "mwCNNUtils.hpp"

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
	size_t maxSize = 0;
	for (auto& lptr : m_layers)
	{
		maxSize = std::max(lptr->GetOutShape().RowCount()
			* lptr->GetOutShape().ColCount() * lptr->OptimizedParamsCount(), maxSize);
	}
	m_workSpace.resize(maxSize, Scalar(0));
	for (auto& lptr : m_layers)
	{
		const size_t dataSize = lptr->OptimizedParamsCount();
		if (dataSize == 0)
			continue;
		lptr->MapData(
			&m_weightsSpace[curMapIdx], &m_gradsSpace[curMapIdx], &m_workSpace[0]);
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
mwTensorView<Scalar> mwCNN<Scalar>::Predict(const mwTensor<Scalar>& x) const
{
	ValidateFinalization();
	m_layers[0]->Forward(x.ToView());
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
void mwCNN<Scalar>::Fit(const std::vector<mwTensor<Scalar>>& x,
	const std::vector<mwTensor<Scalar>>& y,
	std::shared_ptr<mwOptimizer<Scalar>> opt,
	std::shared_ptr<mwLossFunction<Scalar>> loss,
	const size_t epochCount /*= 1*/,
	const size_t batchSize /*= 1*/)
{
	std::vector<mwTensorView<Scalar>> xv;
	std::vector<mwTensorView<Scalar>> yv;
	dmReader::ConvertToTensorView(x, y, xv, yv);
	for (size_t ep = 0; ep < epochCount; ++ep)
	{
		Scalar err = 0;
		size_t steps = 0;
		for (size_t start = 0, end = xv.size(), next = 0;
			start < end; start = next)
		{
			next = std::min(start + batchSize, end);
			const Scalar currentLoss = TrainBatch(opt, loss, xv, yv, start, next);
			err = mwCNNUtils::MovingAverage<Scalar>(err, steps + 1, currentLoss);
			std::cout << "epoch " << ep  << " Batch " << steps << " loss = " << currentLoss << std::endl;
			++steps;
		}
		std::cout << "epoch " << ep << " loss=" << err / Scalar(steps) << std::endl;
		/*if(ep % 1000 == 0)
		std::cout << "epoch " << ep << " loss=" << err / Scalar(steps) << std::endl;*/
	}
}

template<typename Scalar>
void mwCNN<Scalar>::Load(const std::string& filePath)
{
	std::ifstream stream(filePath.c_str(), std::ios::binary);
	std::vector<char> allSymbols;
	
	while (!stream.eof())
	{
		char c;
		stream.read((char*)(&c), 1);
		allSymbols.push_back(c);
	}
	dmBinIStream binStream(allSymbols);
	size_t layersSize;
	binStream >> layersSize;
	for (size_t i = 0; i < layersSize; ++i)
	{
		int typeInt;
		binStream >> typeInt;
		layers::mwLayerType type(
			static_cast<layers::mwLayerType>(typeInt));
		this->AddLayer(serializer::DeserializeLayer(binStream, type, m_layers));
	}
	this->Finalize();
	size_t wSize;
	binStream >> wSize;
	if (wSize != m_weightsSpace.size())
		throw std::exception("Invaild size");
	for (size_t i = 0; i < wSize; ++i)
	{
		binStream >> m_weightsSpace[i];
	}
	for (size_t i = 0; i < wSize; ++i)
	{
		binStream >> m_gradsSpace[i];
	}
}

template<typename Scalar>
void mwCNN<Scalar>::Save(const std::string& filePath)
{
	dmBinOStream binStream;
	binStream << m_layers.size();
	for (size_t i = 0; i < m_layers.size(); ++i)
	{
		binStream << int(m_layers[i]->GetType());
		serializer::SerializeLayer(binStream,
			m_layers[i]->GetType(),*m_layers[i], m_layers);
	}
	binStream << m_weightsSpace.size();
	for (size_t i = 0; i < m_weightsSpace.size(); ++i)
	{
		binStream << m_weightsSpace[i];
	}
	for (size_t i = 0; i < m_gradsSpace.size(); ++i)
	{
		binStream << m_gradsSpace[i];
	}
	std::ofstream stream(filePath.c_str(),
		 std::ios::binary );
	for (char c : binStream.m_values)
	{
		stream << c;
	}
	stream.close();
}


template<typename Scalar>
std::vector<std::shared_ptr<layers::mwLayer<Scalar>>> mwCNN<Scalar>::Layers()
{
	return m_layers;
}


template struct mwCNN<double>;
template struct mwCNN<float>;
