#pragma once
#include <vector>
#include <random>
#include "dmMatrix.hpp"
#include <functional>
#include "dmOptimizer.hpp"
#include <memory>
#include <iostream>
#include "dmLayer.hpp"
#include "dmLayers.hpp"
#include <fstream>
#include "dmDataCollector.hpp"
#include <iomanip>

namespace dmNeural
{
struct dmInOut
{
	std::vector<double> m_in;
	std::vector<double> m_out;
};

typedef dmDataCollector DataCol;
std::unique_ptr<dmLayer> GetLayerByType(const dmLayerType& type, const size_t inS, 
	const size_t outS, const size_t kernelSize)
{
	switch (type)
	{
	case dmLayerType::BIAS: return std::unique_ptr<dmLayer>(new dmBiasLayer(inS));
	case dmLayerType::FULLY_CONNECTED: return std::unique_ptr<dmLayer>(new dmFullyConnectedLayer(inS, outS));
	case dmLayerType::LINEAR_MULT: return std::unique_ptr<dmLayer>(new dmLinearMultLayer(inS, 1));
	case dmLayerType::OUTPUT: return std::unique_ptr<dmLayer>(new dmOutputLayer(inS, {0.}));
	case dmLayerType::RE_LU: return std::unique_ptr<dmLayer>(new dmReLULayer(inS));
	case dmLayerType::TANH: return std::unique_ptr<dmLayer>(new dmHyperbolicTan(inS));
	case dmLayerType::CONVOLUTION: return std::unique_ptr<dmLayer>(new dmConvolutionalLayer(inS, outS, kernelSize));
	default:
		return nullptr;
	}
}

class dmNeuralNetwork
{
public:
	dmNeuralNetwork(const std::function<double()>& randomGen)
		:m_randomGen(randomGen), m_opt(0.000001, 0.6, 0.001, 1), m_isFinalized(false)
	{
	}

	void AddLayer(std::unique_ptr<dmLayer>&& l)
	{
		m_isFinalized = false;
		const size_t capps = l->DataCapasity();
		m_weightsSpace.resize(m_weightsSpace.size() + capps);
		m_gradsSpace.resize(m_gradsSpace.size() + capps);
		m_layers.push_back(std::move(l));
	}

	void Finalize()
	{
		size_t curMapIdx = 0;
		for (std::unique_ptr<dmLayer>& lptr : m_layers)
		{
			const size_t dataSize = lptr->DataCapasity();
			if(dataSize == 0)
				continue;
			lptr->MapData(
				&m_weightsSpace[curMapIdx], &m_gradsSpace[curMapIdx], dataSize);
			curMapIdx += dataSize;
		}
		for (double& w : m_weightsSpace)
		{
			w = m_randomGen();
		}
		m_isFinalized = true;
		m_opt = dmOptimizer::dmOptimizerGradientDescent(
			m_opt.m_learningRate, m_opt.m_momentum, m_opt.m_weightDecay, m_weightsSpace.size());
	}

	std::vector<double> CostGrad(const dmInOut& singleCase)
	{
		if (!m_isFinalized)
			throw std::exception("should be finalized");
		CalcCost(singleCase.m_in);
		BackProp(singleCase);
		CalcGrads(singleCase);
		return m_gradsSpace;
	}

	double CostGradNumeric(dmInOut singleCase, size_t idx, double h)
	{
		if (!m_isFinalized)
			throw std::exception("should be finalized");
		double saved = m_weightsSpace[idx];
		m_weightsSpace[idx] = saved + h;
		double costFwd = this->CalcCost(singleCase.m_in)[0];
		m_weightsSpace[idx] = saved - h;
		double costBck = this->CalcCost(singleCase.m_in)[0];
		m_weightsSpace[idx] = saved;
		return (costFwd - costBck) / h / 2;
	}

	std::vector<double> CostGradNumeric(const dmInOut& singleCase)
	{
		if (!m_isFinalized)
			throw std::exception("should be finalized");
		std::vector<double> gradNum(m_weightsSpace.size());
		for (size_t i = 0; i < gradNum.size(); i++) {
			gradNum[i] = this->CostGradNumeric(singleCase, i, 1e-8);
		}
		return gradNum;
	}

	void BackProp(const dmInOut& item)
	{
		this->CalcCost(item.m_in);
		m_layers.back()->SetOut(item.m_out);
		m_layers.back()->Backprop((*std::prev(m_layers.end(), 2))->Output(),
			m_layers.back()->GetDerivatives());
		for (int i = m_layers.size() - 2; i >= 1; i--)
		{
			m_layers[i]->Backprop(
				m_layers[i - 1]->Output(), m_layers[i + 1]->GetDerivatives());
		}
	}

	void CalcGrads(const dmInOut& item)
	{
		m_layers.back()->CalcGrads((*std::prev(m_layers.end(),2))->Output(),
			m_layers.back()->GetDerivatives());
		for (int i = m_layers.size() - 2; i >= 1; i--)
		{
			m_layers[i]->CalcGrads(
				m_layers[i - 1]->Output(), m_layers[i + 1]->GetDerivatives());
		}
		m_layers[0]->CalcGrads(item.m_in, m_layers[1]->GetDerivatives());
	}

	void TrainBatch(const std::vector<dmInOut>& cases,
		const size_t start, const size_t end, DataCol& collector)
	{
		if (!m_isFinalized)
			throw std::exception("should be finalized");
		std::vector<double> gradSum(m_weightsSpace.size(), 0);
		for (size_t c = start; c < end; ++c)
		{
			BackProp(cases[c]);
			CalcGrads(cases[c]);
			for (size_t i = 0; i < m_gradsSpace.size(); ++i)
			{
				gradSum[i] += m_gradsSpace[i];
			}			
		}
		const size_t bSize = (end - start);
		for (size_t i = 0; i < gradSum.size(); ++i)
		{
			gradSum[i] /= double(bSize);
		}
		m_opt.Update(gradSum, m_weightsSpace);
	}

	double CalcError(const dmInOut& item)
	{
		const std::vector<double>& vals = this->Predict(item.m_in);
		double res = 0;
		for (size_t i = 0; i < vals.size(); ++i)
		{
			res += (item.m_out[i] - vals[i]) * (item.m_out[i] - -vals[i]);
		}
		return res;
	}

	double CalcError(const std::vector<dmInOut>& items)
	{
		double err = 0;
		for (auto& it : items)
		{
			err += CalcError(it);
		}
		return err / double(items.size());
	}

	double Train(const size_t epochCount, const size_t batchSize, const std::vector<dmInOut>& cases,
		DataCol& collector)
	{
		if (!m_isFinalized)
			throw std::exception("should be finalized");
		for (size_t ep = 0; ep < epochCount; ++ep)
		{
			for (size_t start = 0, end = cases.size(), next =0;
				start < end; start = next)
			{
				next = std::min(start + batchSize, end);
				TrainBatch(cases, start, next, collector);
			}
			
			if (ep % 1000 != 0)
				continue;
			const double err = CalcError(cases);
			collector.Collect(err);
			std::cout << "case " << ep << " err=" << err << std::endl;
		}
		return 0;
	}

	std::vector<double> CalcCost(const std::vector<double>& in)
	{
		if (!m_isFinalized)
			throw std::exception("should be finalized");
		for (int i = 0; i < m_layers.size(); i++)
		{
			if (i == 0)
				m_layers[i]->Forward(in);
			else
				m_layers[i]->Forward(m_layers[i - 1]->Output());
		}
		return m_layers.back()->Output();
	}

	std::vector<double> Predict(const std::vector<double>& in)
	{
		if (!m_isFinalized)
			throw std::exception("should be finalized");
		for (int i = 0; i < m_layers.size(); i++)
		{
			if (i == 0)
				m_layers[i]->Forward(in);
			else
				m_layers[i]->Forward(m_layers[i - 1]->Output());
		}
		return (*std::prev(m_layers.end(), 2))->Output();
	}

	std::string Serialize()
	{
		if (!m_isFinalized)
			throw std::exception("should be finalized");

		const std::string spliter = " ";
		std::stringstream ss;
		ss << m_opt.m_learningRate << spliter
			<< m_opt.m_momentum << spliter
			<< m_opt.m_weightDecay << spliter
			<< m_layers.size() << spliter << std::endl;
		for (auto& l : m_layers)
		{
			ss << int(l->GetType()) << std::endl;
			
			if (l->GetType() == dmLayerType::CONVOLUTION)
			{
				dmConvolutionalLayer* conv = dynamic_cast<dmConvolutionalLayer*>(l.get());
				ss << conv->GetImageW() << spliter << conv->GetImageH()
					<< spliter<< conv->GetKernelSize();
			}
			else
			{
				ss << l->GetInSize() << spliter << l->GetOutSize();
			}
			ss << std::endl;
		}
		ss << m_weightsSpace.size() << std::endl;
		for (size_t i = 0; i < m_weightsSpace.size(); ++i)
		{
			ss << m_weightsSpace[i] << spliter;
		}
		return ss.str();
	}

	bool SaveModel(const std::string& pathToFile)
	{
		if (!m_isFinalized)
			throw std::exception("should be finalized");
		std::ofstream ofs(pathToFile);
		if (!ofs.is_open())
		{
			return false;
		}
		ofs << std::fixed << std::setprecision(9);
		ofs << Serialize();
		return true;
	}

	bool LoadModel(const std::string& pathToFile)
	{
		m_isFinalized = false;
		m_layers.clear();
		m_weightsSpace.clear();
		m_gradsSpace.clear();
		std::ifstream ifs(pathToFile);
		if (!ifs.is_open())
		{
			return false;
		}
		std::string line;
		std::getline(ifs, line);
		std::stringstream modelParams(line);
		modelParams >> m_opt.m_learningRate;
		modelParams >> m_opt.m_momentum;
		modelParams >> m_opt.m_weightDecay;
		size_t layerCount;
		modelParams >> layerCount;
		for (size_t i = 0; i < layerCount; ++i)
		{
			std::getline(ifs, line);
			std::stringstream lType(line);
			int lTypeInt;
			lType >> lTypeInt;
			std::getline(ifs, line);
			std::stringstream lSyze(line);
			size_t inS, outS, kernel = 0;
			lSyze >> inS >> outS;
			if (dmLayerType(lTypeInt) == dmLayerType::CONVOLUTION)
			{
				lSyze >> kernel;
			}
			this->AddLayer(GetLayerByType(dmLayerType(lTypeInt), inS, outS, kernel));
		}
		Finalize();
		std::getline(ifs, line);
		std::stringstream wS(line);
		size_t wCount;
		wS >> wCount;
		if (wCount != m_weightsSpace.size())
			throw std::exception("invalid desirialize");
		std::getline(ifs, line);
		std::stringstream wVal(line);
		for (size_t i = 0; i < wCount; ++i)
		{
			wVal >> m_weightsSpace[i];
		}
	}

	std::vector<std::unique_ptr<dmLayer>> m_layers;
	dmOptimizer::dmOptimizerGradientDescent m_opt;
	std::vector<double> m_weightsSpace;
	std::vector<double> m_gradsSpace;
	const std::function<double()> m_randomGen;
	bool m_isFinalized;
};

}