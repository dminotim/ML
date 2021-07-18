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
#include <iomanip>
#include "dmRGBImage.hpp"
#include "dmPainterIO.hpp"
#include "mwTensor.hpp"
#include <random>
#include "mwCNN.hpp"
#include "mwAdamOptimizer.hpp"
#include "mwMSELossFunction.hpp"
#include "mwConv2dLayer.hpp"
#include "mwReluLayer.hpp"
#include "mwMaxPoolLayer.hpp"
#include "mwFCLayer.hpp"
#include "mwDropOutLayer.hpp"
#include "mwSoftMax.hpp"
#include "mwCrossEntropyLossFunction.hpp"


int main()
{
	mwCNN<float> cnn;
	std::shared_ptr<mwOptimizer<float>> optimizer = std::make_shared<mwAdamOptimizer<float>>();

	std::shared_ptr<mwLossFunction<float>> lossMse = std::make_shared<mwCrossEntropyLossFunction<float>>();

	{

		auto mnist = dmReader::DownloadMNIST<float>("C:\\projects\\MyML\\mnist_png\\training\\");
		unsigned seed = 234324;

		std::shuffle(mnist.begin(), mnist.end(), std::default_random_engine(seed));
		std::vector<mwTensor<float>> tesvecX;
		std::vector<mwTensor<float>> tesvecY;
		std::vector<mwTensorView<float>> tesvecVX;
		std::vector<mwTensorView<float>> tesvecVY;
		dmReader::ConvertMNISTToTensors(mnist, tesvecX, tesvecY);
		dmReader::ConvertToTensorView(tesvecX, tesvecY, tesvecVX, tesvecVY);



		auto conv1 = std::make_shared<layers::mwConv2dLayer<float>>(16, 3, tesvecVX.back());
		cnn.AddLayer(conv1);
		cnn.AddLayer(std::make_shared<layers::mwReluLayer<float>>(conv1->GetOutShape()));
		std::cout << "Prams size " << conv1->OptimizedParamsCount()
			<< " shape " << conv1->GetOutShape().RowCount() << " "
			<< conv1->GetOutShape().ColCount() << " "
			<< conv1->GetOutShape().Depth() << std::endl;
		auto maxp = std::make_shared<layers::mwMaxPoolLayer<float>>(2, conv1->GetOutShape());
		cnn.AddLayer(maxp);
		auto conv2 = std::make_shared<layers::mwConv2dLayer<float>>(32, 3, maxp->GetOutShape());
		cnn.AddLayer(conv2);

		std::cout << "Prams size " << conv2->OptimizedParamsCount()
			<< " shape " << conv2->GetOutShape().RowCount() << " "
			<< conv2->GetOutShape().ColCount() << " "
			<< conv2->GetOutShape().Depth() << std::endl;

		auto maxp2 = std::make_shared<layers::mwMaxPoolLayer<float>>(2, conv2->GetOutShape());
		cnn.AddLayer(maxp2);
		std::cout << "Prams size " << maxp2->OptimizedParamsCount()
			<< " shape " << maxp2->GetOutShape().RowCount() << " "
			<< maxp2->GetOutShape().ColCount() << " "
			<< maxp2->GetOutShape().Depth() << std::endl;

		cnn.AddLayer(std::make_shared<layers::mwReluLayer<float>>(maxp2->GetOutShape()));

		cnn.AddLayer(std::make_shared<layers::mwDropOutLayer<float>>(maxp2->GetOutShape()));

		//auto fcLayer1 =
		//	std::make_shared<layers::mwFCLayer<float>>(10, maxp2->GetOutShape());
		//cnn.AddLayer(fcLayer1);
		auto fcLayer2 = std::make_shared<layers::mwFCLayer<float>>(10, maxp2->GetOutShape());
		cnn.AddLayer(fcLayer2);

		std::cout << "Prams size " << fcLayer2->OptimizedParamsCount()
			<< " shape " << fcLayer2->GetOutShape().RowCount() << " "
			<< fcLayer2->GetOutShape().ColCount() << " "
			<< fcLayer2->GetOutShape().Depth() << std::endl;
		cnn.AddLayer(std::make_shared<layers::mwSoftMax<float>>(fcLayer2->GetOutShape()));
		//auto fcLayer3= std::make_shared<layers::mwFCLayer<double>>(1, fcLayer2->GetOutShape());
		//cnn.AddLayer(fcLayer3);
		//cnn.AddLayer(std::make_shared<layers::mwReluLayer<double>>(fcLayer3->GetOutShape()));
		cnn.Finalize();
		std::cout << "Fit" << std::endl;
		cnn.Fit(tesvecVX, tesvecVY, optimizer, lossMse, 5, 1);
		std::cout << "Fit 2" << std::endl;
	}
	auto mnist = dmReader::DownloadMNIST<float>("C:\\projects\\MyML\\mnist_png\\testing\\");
	unsigned seed = 234324;

	std::shuffle(mnist.begin(), mnist.end(), std::default_random_engine(seed));
	std::vector<mwTensor<float>> tesvecX;
	std::vector<mwTensor<float>> tesvecY;
	std::vector<mwTensorView<float>> tesvecVX;
	std::vector<mwTensorView<float>> tesvecVY;
	dmReader::ConvertMNISTToTensors(mnist, tesvecX, tesvecY);
	dmReader::ConvertToTensorView(tesvecX, tesvecY, tesvecVX, tesvecVY);

	int goodCount = 0;
	for (int i = 0; i < tesvecVX.size(); ++i)
	{
		auto pred = cnn.Predict(tesvecVX[i]);
		size_t idx1 = dmReader::GetMaxIndex(pred);
		size_t idx2 = dmReader::GetMaxIndex(tesvecVY[i]);

		if (idx1 == idx2)
		{
			goodCount++;
		}
	}

	std::cout << "accuracy = " << double(goodCount) / double(tesvecVX.size()) << std::endl;
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
