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
#include "dmBinaryStream.hpp"
#include "mwZeroPaddingLayer.hpp"
#include "mwUpsamplingLayer.hpp"
#include "mwConcatLayer.hpp"
#include "mwSigmoid.hpp"
#include "mwUnetCreator.hpp"
#include "mwCNNUtils.hpp"



int main()
{
	dmBinOStream out;
	out << size_t(124);
	out << size_t(126);
	dmBinIStream inStr(out.m_values);
	size_t a;
	size_t b;
	inStr >> a >> b;
	std::cout << std::fixed << std::setprecision(14) <<  a <<" " <<  b << std::endl;
	mwCNN<float> cnn;
	std::shared_ptr<mwOptimizer<float>> optimizer = std::make_shared<mwAdamOptimizer<float>>();

	std::shared_ptr<mwLossFunction<float>> lossMse = std::make_shared<mwCrossEntropyLossFunction<float>>();

	{
	/*	auto mnist = dmReader::DownloadMNIST<float>("C:\\projects\\MyML\\mnist_png\\training\\");
		unsigned seed = 234324;
		std::shuffle(mnist.begin(), mnist.end(), std::default_random_engine(seed));*/
		/*
		layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
		layers.MaxPooling2D(pool_size=(2, 2)),
		layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
		layers.MaxPooling2D(pool_size=(2, 2)),
		layers.Flatten(),
		layers.Dropout(0.5),
		layers.Dense(num_classes, activation="softmax"),
		*/
		
	/*	std::vector<mwTensor<float>> tesvecX;
		std::vector<mwTensor<float>> tesvecY;
		dmReader::ConvertMNISTToTensors(mnist, tesvecX, tesvecY);*/
		std::vector<mwTensor<float>> tesvecX;
		std::vector<mwTensor<float>> tesvecY;
		dmReader::DownloadXYImage("C:\\projects\\paint_by_number\\cases2\\input\\",
			"C:\\projects\\paint_by_number\\cases2\\output\\",
			tesvecX, tesvecY);

		mwTensorView<float> inputShape(nullptr, 256, 256, 1);
		//mwTensorView<float> inputShape(nullptr, 28, 28, 1);
		//cnn.AddLayer(std::make_shared<layers::mwConv2dLayer<float>>(32, 3, inputShape));
		//cnn.AddLayer(std::make_shared<layers::mwReluLayer<float>>(cnn.Layers().back()->GetOutShape()));
		//cnn.AddLayer(std::make_shared<layers::mwMaxPoolLayer<float>>(2, cnn.Layers().back()->GetOutShape()));
		//cnn.AddLayer(std::make_shared<layers::mwConv2dLayer<float>>(64, 3, cnn.Layers().back()->GetOutShape()));
		//cnn.AddLayer(std::make_shared<layers::mwReluLayer<float>>(cnn.Layers().back()->GetOutShape()));
		//cnn.AddLayer(std::make_shared<layers::mwMaxPoolLayer<float>>(2, cnn.Layers().back()->GetOutShape()));
		//cnn.AddLayer(std::make_shared<layers::mwDropOutLayer<float>>(cnn.Layers().back()->GetOutShape(), float(0.5)));
		//cnn.AddLayer(std::make_shared<layers::mwFCLayer<float>>(10, cnn.Layers().back()->GetOutShape()));
		//cnn.AddLayer(std::make_shared<layers::mwSoftMax<float>>(cnn.Layers().back()->GetOutShape()));
		//cnn.Finalize();
		//
		mwUnetCreator::Create(inputShape, cnn);
		std::cout << "Fit" << std::endl;
		auto b = std::chrono::high_resolution_clock::now();
		cnn.Fit(tesvecX, tesvecY, optimizer, lossMse, 1, 5);
		auto e = std::chrono::high_resolution_clock::now();
		std::cout << "time = " << 
			double(std::chrono::duration_cast<std::chrono::seconds>(e - b).count())/60. << std::endl;
		std::cout << "Fit 2" << std::endl;
	}
	
	/*auto mnist = dmReader::DownloadMNIST<float>("C:\\projects\\MyML\\mnist_png\\testing\\");
	unsigned seed = 234324;

	std::shuffle(mnist.begin(), mnist.end(), std::default_random_engine(seed));
	std::vector<mwTensor<float>> tesvecX;
	std::vector<mwTensor<float>> tesvecY;
	dmReader::ConvertMNISTToTensors(mnist, tesvecX, tesvecY);*/

	cnn.Save("C:\\projects\\MyML\\model_unet.bin");
	/*int goodCount = 0;
	for (int i = 0; i < tesvecX.size(); ++i)
	{
		auto pred = cnn.Predict(tesvecX[i]);
		size_t idx1 = dmReader::GetMaxIndex(pred);
		size_t idx2 = dmReader::GetMaxIndex(tesvecY[i].ToView());
		if (idx1 == idx2)
		{
			goodCount++;
		}
	}
	std::cout << "accuracy = " << double(goodCount) / double(tesvecX.size()) << std::endl;*/

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
