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

std::shared_ptr<layers::mwLayer<float>> AddConvolution(mwCNN<float>& cnn, const size_t fc,
	const mwTensorView<float>& inShape)
{
	auto zeroP = std::make_shared<layers::mwZeroPaddingLayer<float>>(1, inShape);
	cnn.AddLayer(zeroP);
	auto convolv = std::make_shared<layers::mwConv2dLayer<float>>(fc, 3, zeroP->GetOutShape());
	cnn.AddLayer(convolv);
	auto relu = std::make_shared<layers::mwReluLayer<float>>(convolv->GetOutShape());
	cnn.AddLayer(relu);
	return convolv;
}

std::shared_ptr<layers::mwLayer<float>> AddUpsampling(mwCNN<float>& cnn, const size_t fc,
	const mwTensorView<float>& inShape)
{
	auto up1 = std::make_shared<layers::mwUpsamplingLayer<float>>(2, inShape);
	cnn.AddLayer(up1);
	auto zeroP = std::make_shared<layers::mwZeroPaddingLayer<float>>(1, up1->GetOutShape());
	cnn.AddLayer(zeroP);
	auto convolv = std::make_shared<layers::mwConv2dLayer<float>>(fc, 3, zeroP->GetOutShape());
	cnn.AddLayer(convolv);
	auto relu = std::make_shared<layers::mwReluLayer<float>>(convolv->GetOutShape());
	cnn.AddLayer(relu);
	return convolv;
}

std::shared_ptr<layers::mwLayer<float>> AddPooling(
	mwCNN<float>& cnn,
	const mwTensorView<float>& inShape)
{
	auto maxP = std::make_shared<layers::mwMaxPoolLayer<float>>(2, inShape);
	cnn.AddLayer(maxP);
	return maxP;
}

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

		/*auto mnist = dmReader::DownloadMNIST<float>("C:\\projects\\MyML\\mnist_png\\training\\");
		unsigned seed = 234324;
		std::shuffle(mnist.begin(), mnist.end(), std::default_random_engine(seed));
		
		dmReader::ConvertMNISTToTensors(mnist, tesvecX, tesvecY);*/
		std::vector<mwTensor<float>> tesvecX;
		std::vector<mwTensor<float>> tesvecY;
		dmReader::DownloadXYImage("C:\\projects\\paint_by_number\\cases2\\input\\",
			"C:\\projects\\paint_by_number\\cases2\\output\\",
			tesvecX, tesvecY);

		mwTensorView<float> inputShape(nullptr, 256, 256, 1);
		auto conv1 = AddConvolution(cnn, 64, inputShape); size_t c1Idx = cnn.Layers().size() - 1;
		auto pool1 = AddPooling(cnn, conv1->GetOutShape());
		auto conv2 = AddConvolution(cnn, 128, pool1->GetOutShape()); size_t c2Idx = cnn.Layers().size() - 1;
		auto pool2 = AddPooling(cnn, conv2->GetOutShape());
		auto conv3 = AddConvolution(cnn, 256, pool2->GetOutShape()); size_t c3Idx = cnn.Layers().size() - 1;
		auto pool3 = AddPooling(cnn, conv3->GetOutShape());
		auto conv4 = AddConvolution(cnn, 512, pool3->GetOutShape()); size_t c4Idx = cnn.Layers().size() - 1;
		cnn.AddLayer(std::make_shared<layers::mwDropOutLayer<float>>(conv4->GetOutShape())); size_t d4 = cnn.Layers().size() - 1;
		auto pool4 = AddPooling(cnn, conv4->GetOutShape());
		auto conv5= AddConvolution(cnn, 1024, pool4->GetOutShape()); size_t c5Idx = cnn.Layers().size() - 1;
		cnn.AddLayer(std::make_shared<layers::mwDropOutLayer<float>>(conv5->GetOutShape())); size_t d5= cnn.Layers().size() - 1;
		auto up1 = AddUpsampling(cnn, 512, conv5->GetOutShape());
		cnn.AddLayer(std::make_shared<layers::mwConcatLayer<float>>(
			cnn.Layers(), std::vector<size_t>(1, d4), up1->GetOutShape()));
		auto merge1 = cnn.Layers().back();
		auto conv6 = AddConvolution(cnn, 512, merge1->GetOutShape());
		auto up2 = AddUpsampling(cnn, 256, conv6->GetOutShape());
		cnn.AddLayer(std::make_shared<layers::mwConcatLayer<float>>(
			cnn.Layers(), std::vector<size_t>(1, c3Idx), up2->GetOutShape()));
		auto conv7 = AddConvolution(cnn, 256, cnn.Layers().back()->GetOutShape());
		auto up3 = AddUpsampling(cnn, 128, conv7->GetOutShape());
		cnn.AddLayer(std::make_shared<layers::mwConcatLayer<float>>(
			cnn.Layers(), std::vector<size_t>(1, c2Idx), up3->GetOutShape()));
		auto conv8 = AddConvolution(cnn, 128, cnn.Layers().back()->GetOutShape());
		auto up4 = AddUpsampling(cnn, 64, conv8->GetOutShape());
		cnn.AddLayer(std::make_shared<layers::mwConcatLayer<float>>(
			cnn.Layers(), std::vector<size_t>(1, c1Idx), up4->GetOutShape()));

		auto conv9 = AddConvolution(cnn, 64, cnn.Layers().back()->GetOutShape());
		auto conv10 = AddConvolution(cnn, 64, cnn.Layers().back()->GetOutShape());
		auto conv11 = AddConvolution(cnn, 2, cnn.Layers().back()->GetOutShape());
		auto conv12 = AddConvolution(cnn, 1, cnn.Layers().back()->GetOutShape());
		cnn.AddLayer(std::make_shared<layers::mwSigmoid<float>>(conv12->GetOutShape()));
		cnn.Finalize();

		std::cout << "Fit" << std::endl;
		cnn.Fit(tesvecX, tesvecY, optimizer, lossMse, 1, 1);
		std::cout << "Fit 2" << std::endl;
	}
	
	auto mnist = dmReader::DownloadMNIST<float>("C:\\projects\\MyML\\mnist_png\\testing\\");
	unsigned seed = 234324;

	std::shuffle(mnist.begin(), mnist.end(), std::default_random_engine(seed));
	std::vector<mwTensor<float>> tesvecX;
	std::vector<mwTensor<float>> tesvecY;
	dmReader::ConvertMNISTToTensors(mnist, tesvecX, tesvecY);

	cnn.Save("C:\\projects\\MyML\\model.bin");
	int goodCount = 0;
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
	std::cout << "accuracy = " << double(goodCount) / double(tesvecX.size()) << std::endl;

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
