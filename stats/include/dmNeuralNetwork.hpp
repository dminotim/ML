#pragma once
#include <vector>
#include <random>
#include "dmMatrix.hpp"
#include <functional>
#include "dmOptimizer.hpp"
#include <memory>
#include <iostream>
#include "dmLayer.hpp"

namespace dmNeural
{
struct dmInOut
{
	std::vector<double> m_in;
	std::vector<double> m_out;
};

class dmNeuralNetwork
{
public:
	dmNeuralNetwork(const std::function<double()>& randomGen)
		:m_randomGen(randomGen), m_opt(0.00005, 0.6, 0.001)
	{
	}
	void AddLayer(std::unique_ptr<dmLayer>&& l)
	{
		l->Init(m_randomGen);
		m_layers.push_back(std::move(l));
	}

	double Train(const dmInOut& singleCase)
	{
		for (int i = 0; i < m_layers.size(); i++)
		{
			if (i == 0)
				m_layers[i]->Forward(singleCase.m_in);
			else
				m_layers[i]->Forward(m_layers[i-1]->Output());
		}
		m_layers.back()->SetOut(singleCase.m_out);
		std::vector<double> temp;
		for (int i = m_layers.size() - 1; i >= 0; i--)
		{
			if (i == 0)
			{
				m_layers[i]->Backprop(singleCase.m_in, m_layers[i + 1]->GetDerivatives());
				continue;
			}
			if (i == m_layers.size() - 1)
				m_layers[i]->Backprop(
					m_layers[i-1]->Output(), temp);
			else
				m_layers[i]->Backprop(
					m_layers[i - 1]->Output(), m_layers[i + 1]->GetDerivatives());
		}

		for (int i = 0; i < m_layers.size(); i++)
		{
			m_layers[i]->Update(m_opt);
		}

		double err = 0;
		int idx = 0;
		for (int i = 0; i < m_layers.size(); i++)
		{
			if (i == 0)
				m_layers[i]->Forward(singleCase.m_in);
			else
				m_layers[i]->Forward(m_layers[i - 1]->Output());
		}
		for (double v : m_layers.back()->Output())
		{
			err += std::abs(singleCase.m_out[idx] - v);
			idx++;
		}
		return err * 100;
	}

	double Train(const size_t epochCount, const std::vector<dmInOut>& cases)
	{
		double amse = 0;
		int index = 0;

		for (long ep = 0; ep < epochCount; ++ep)
		{
			for (const dmInOut& t : cases)
			{
				double xerr = Train(t);
				amse += xerr;
				index++;
			}
			if (ep % 1000 == 0)
				std::cout << "case " << ep << " err=" << amse / index << std::endl;
		}
		return amse / index;
	}

	std::vector<double> Guess(const std::vector<double>& in)
	{
		for (int i = 0; i < m_layers.size(); i++)
		{
			if (i == 0)
				m_layers[i]->Forward(in);
			else
				m_layers[i]->Forward(m_layers[i - 1]->Output());
		}
		return m_layers.back()->Output();
	}
	std::vector<std::unique_ptr<dmLayer>> m_layers;
	dmOptimizer::dmOptimizerGradientDescent m_opt;
	const std::function<double()> m_randomGen;
};

}