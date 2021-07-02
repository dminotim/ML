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

namespace dmNeural
{
struct dmInOut
{
	std::vector<double> m_in;
	std::vector<double> m_out;
};


std::unique_ptr<dmLayer> GetLayerByType(const dmLayerType& type, const size_t inS, 
	const size_t outS)
{
	switch (type)
	{
	case dmLayerType::BIAS: return std::unique_ptr<dmLayer>(new dmBiasLayer(inS));
	case dmLayerType::FULLY_CONNECTED: return std::unique_ptr<dmLayer>(new dmFullyConnectedLayer(inS, outS));
	case dmLayerType::LINEAR_MULT: return std::unique_ptr<dmLayer>(new dmLinearMultLayer(inS, 1));
	case dmLayerType::OUTPUT: return std::unique_ptr<dmLayer>(new dmOutputLayer(inS, {0.}));
	case dmLayerType::RE_LU: return std::unique_ptr<dmLayer>(new dmReLULayer(inS));
	case dmLayerType::TANH: return std::unique_ptr<dmLayer>(new dmHyperbolicTan(inS));
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
		for (int i = 0; i < m_layers.size(); i++)
		{
			if (i == 0)
				m_layers[i]->Forward(singleCase.m_in);
			else
				m_layers[i]->Forward(m_layers[i - 1]->Output());
		}
		m_layers.back()->SetOut(singleCase.m_out);
		std::vector<double> temp;
		for (int i = m_layers.size() - 1; i >= 0; i--)
		{
			if (i == 0)
			{
				m_layers[i]->Backprop(singleCase.m_in, m_layers[i + 1]->GetDerivatives());
				m_layers[i]->CalcGrads(singleCase.m_in, m_layers[i + 1]->GetDerivatives());
				continue;
			}
			if (i == m_layers.size() - 1)
			{
				m_layers[i]->Backprop(
					m_layers[i - 1]->Output(), temp);
				m_layers[i]->CalcGrads(
					m_layers[i - 1]->Output(), temp);
			}
			else
			{
				m_layers[i]->Backprop(
					m_layers[i - 1]->Output(), m_layers[i + 1]->GetDerivatives());
				m_layers[i]->CalcGrads(
					m_layers[i - 1]->Output(), m_layers[i + 1]->GetDerivatives());
			}
		}
		return m_gradsSpace;
	}
	double CostGradNumeric(dmInOut singleCase, size_t idx, double h)
	{
		if (!m_isFinalized)
			throw std::exception("should be finalized");
		double saved = m_weightsSpace[idx];
		m_weightsSpace[idx] = saved + h;
		double costFwd = this->Guess(singleCase.m_in)[0];
		m_weightsSpace[idx] = saved - h;
		double costBck = this->Guess(singleCase.m_in)[0];
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

	// end not included
	double TrainBatch(const std::vector<dmInOut>& cases,
		const size_t start, const size_t end)
	{
		if (!m_isFinalized)
			throw std::exception("should be finalized");
		std::vector<double> gradSum(m_weightsSpace.size(), 0);
		for (size_t c = start; c < end; ++c)
		{
			this->Guess(cases[c].m_in);

			m_layers.back()->SetOut(cases[c].m_out);
			std::vector<double> temp;
			for (int i = m_layers.size() - 1; i >= 0; i--)
			{
				if (i == 0)
				{
					m_layers[i]->Backprop(cases[c].m_in, m_layers[i + 1]->GetDerivatives());
					continue;
				}
				if (i == m_layers.size() - 1)
				{
					m_layers[i]->Backprop(
						m_layers[i - 1]->Output(), temp);
				}
				else {
					m_layers[i]->Backprop(
						m_layers[i - 1]->Output(), m_layers[i + 1]->GetDerivatives());
				}
			}
			for (int i = m_layers.size() - 1; i >= 0; i--)
			{
				if (i == 0)
				{
					m_layers[i]->CalcGrads(cases[c].m_in, m_layers[i + 1]->GetDerivatives());
					continue;
				}
				if (i == m_layers.size() - 1)
				{
					m_layers[i]->CalcGrads(
						m_layers[i - 1]->Output(), temp);
				}
				else {
					m_layers[i]->CalcGrads(
						m_layers[i - 1]->Output(), m_layers[i + 1]->GetDerivatives());
				}
			}
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
		double err = 0;
		return err;
	}

	double Train(const size_t epochCount, const size_t batchSize, const std::vector<dmInOut>& cases)
	{
		if (!m_isFinalized)
			throw std::exception("should be finalized");
	
		int index = 0;


		for (long ep = 0; ep < epochCount; ++ep)
		{
			size_t start = 0;
			size_t end = cases.size();
			
			while (start < end)
			{
				const size_t next = std::min(start + batchSize, end);
				TrainBatch(cases, start, next);
				start = next;
			}
			
			if (ep % 1000 == 0)
			{
				double amse = 0;
				for (auto& c : cases)
				{
					int idx = 0;
					for (double v : this->Predict(c.m_in))
					{
						amse += std::abs(c.m_out[idx] - v);
						idx++;
					}
				}
				std::cout << "case " << ep << " err=" << amse / double(cases.size()) << std::endl;
			}
		}
		return 0;
	}

	std::vector<double> Guess(const std::vector<double>& in)
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
			ss << l->GetInSize() << spliter<< l->GetOutSize() << std::endl;
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
			size_t inS, outS;
			lSyze >> inS >> outS;
			this->AddLayer(GetLayerByType(dmLayerType(lTypeInt), inS, outS));
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