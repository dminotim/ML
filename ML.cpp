// MatrixApp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "dmMatrix.hpp"
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <chrono>
#include "dmLUDecompose.hpp"
#include "dmLDLDecompose.hpp"
#include "dmDataReader.hpp"
#include "dmStatsCore.hpp"
#include <memory>
#include "dmNeuralNetwork.hpp"
#include "dmLayers.hpp"
#include <iomanip>
// check gradient
//vector<double> grad = cnn->cost_grad(input, target);
//vector<double> gradNum(grad.size());
//for (size_t i = 0; i < grad.size(); i++) {
//	gradNum[i] = cnn->cost_grad_numeric(input, target, i, 1e-8);
//}
//double maxDiff = -DBL_MAX;
//for (size_t i = 0; i < grad.size(); i++) {
//	maxDiff = (max)(maxDiff, fabs(grad[i] - gradNum[i]));
//}



//double MeshCNN::cost_grad_numeric(const shared_ptr<const MeshFrame>& inp, const shared_ptr<const MeshFrame>& target, size_t idx, double h)
//{
//	vector<double> params = serialize();
//	double saved = params[idx];
//	params[idx] = saved + h;
//	deserialize(params);
//	double costFwd = cost(inp, target);
//	params[idx] = saved - h;
//	deserialize(params);
//	double costBck = cost(inp, target);
//	params[idx] = saved;
//	deserialize(params);
//	return (costFwd - costBck) / h / 2;
//}

//layers.back()->delta = lastDelta;
//for (auto it = std::prev(layers.end()); it != layers.begin(); --it) {
//	(**it).backprop(**std::prev(it));
//}
//shared_ptr<const MeshFrame> prevFrame = inp;
//for (shared_ptr<Layer> layerPtr : layers) {
//	ComputeGradientsVisitor computeGradients(*prevFrame);
//	layerPtr->accept(computeGradients);
//	prevFrame = layerPtr->out;
//}

int main()
{
	/*auto samples = dmReader::Read("C:\\projects\\data.txt");
	auto covariance = dmStatsCore::GetCorrelationMatrix(samples);
	auto r2 = dmStatsCore::GetR2Matrix(samples);
	std::cout << std::endl;
	covariance.Print();
	std::cout << std::endl;
	std::cout << std::endl;
	r2.Print();*/
	
	std::random_device r;

	// Choose a random mean between 1 and 6
	std::default_random_engine e1(3731);
	std::uniform_real_distribution<double> uniform_dist(0.0, 0.1);
	std::function<double()> rnd = [&]() { return uniform_dist(e1); };
	std::vector<std::vector<double>> xVals;
	
	for (size_t i = 0; i < 100; ++i)
	{
		xVals.push_back({ rnd() * 10., rnd() * 10.,  rnd() * 10. });
	}
	std::vector<double> yVals;
	std::vector<dmNeural::dmInOut> cases;
	for (size_t i = 0; i < xVals.size(); ++i)
	{
		yVals.push_back(xVals[i][0] * xVals[i][0] + xVals[i][1] * xVals[i][1] + xVals[i][2]);
		dmNeural::dmInOut toAdd;
		toAdd.m_in = xVals[i];
		toAdd.m_out = std::vector<double>(1, yVals[i]);
		cases.push_back(toAdd);
	}

	dmNeural::dmNeuralNetwork net(rnd);
	net.AddLayer(std::unique_ptr<dmNeural::dmLayer>(new dmNeural::dmFullyConnectedLayer(3, 20)));
	net.AddLayer(std::unique_ptr<dmNeural::dmLayer>(new dmNeural::dmBiasLayer(20)));
	net.AddLayer(std::unique_ptr<dmNeural::dmLayer>(new dmNeural::dmHyperbolicTan(20)));
	net.AddLayer(std::unique_ptr<dmNeural::dmLayer>(new dmNeural::dmFullyConnectedLayer(20, 1)));
	net.AddLayer(std::unique_ptr<dmNeural::dmLayer>(new dmNeural::dmBiasLayer(1)));
	net.AddLayer(std::unique_ptr<dmNeural::dmLayer>(new dmNeural::dmHyperbolicTan(1)));
	net.AddLayer(std::unique_ptr<dmNeural::dmLayer>(new dmNeural::dmOutputLayer(1, { 0. })));
	//net.AddLayer(std::unique_ptr<dmNeural::dmLayer>(new dmNeural::dmBiasLayer(10)));
	//net.AddLayer(std::unique_ptr<dmNeural::dmLayer>(new dmNeural::dmReLULayer(10)));
	//net.AddLayer(std::unique_ptr<dmNeural::dmLayer>(new dmNeural::dmFullyConnectedLayer(10, 1.)));
	//net.AddLayer(std::unique_ptr<dmNeural::dmLayer>(new dmNeural::dmBiasLayer(1)));
	//net.AddLayer(std::unique_ptr<dmNeural::dmLayer>(new dmNeural::dmReLULayer(1)));

	//net.AddLayer(std::unique_ptr<dmNeural::dmLayer>(new dmNeural::dmOutputLayer(1, { 0. })));
	net.Finalize();
	net.Train(50000, 1, cases);
	//net.SaveModel("C:\\projects\\TEST\\model.txt");
	//net.LoadModel("C:\\projects\\TEST\\model.txt");
	std::cout << std::endl;
	std::cout << std::fixed << std::setprecision(9);
	/*std::vector<double> sumReal();*/
	for (size_t i = 0; i < 1; ++i)
	{
		std::cout << std::endl;
		std::cout << "In: " << cases[i].m_in[0] << ", " << cases[i].m_in[1]
			<< " Out: " << cases[i].m_out[0] << std::endl;
		std::cout << "Ans " << net.Predict(cases[i].m_in)[0] << std::endl;
		for (const double v : net.CostGrad(cases[i]))
		{
			std::cout << v << " ";
		}
		std::cout << std::endl;
		for (const double v : net.CostGradNumeric(cases[i]))
		{
			std::cout << v << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	/*for (size_t i = 0; i < 100; ++i)
	{
		const double val1 = rnd() * 10;
		const double val2 = rnd() * 10;
		std::cout << net.Guess({ val1, val2 })[0] << " : " << val1 * val1 + val2 * val2 << std::endl;
	}*/
	//net.Train()
	//for (auto& m : samples)
	//{
	//	std::cout << "###### " << dmStatsCore::GetMean(m) <<  "\n";
	//	/*for (auto v : m.m_values)
	//	{
	//		std::cout << v << " ";
	//	}*/
	//	std::cout << std::endl;
	//}
	std::cout << "Hello World!\n";
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
