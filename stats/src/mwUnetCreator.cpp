#include "mwUnetCreator.hpp"
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

namespace mwUnetCreator
{

template<class Scalar>
std::shared_ptr<layers::mwLayer<Scalar>> AddConvolution(mwCNN<Scalar>& cnn, const size_t fc,
	const mwTensorView<Scalar>& inShape, bool uselastRelu = true)
{
	auto zeroP = std::make_shared<layers::mwZeroPaddingLayer<Scalar>>(1, inShape);
	cnn.AddLayer(zeroP);
	auto convolv = std::make_shared<layers::mwConv2dLayer<Scalar>>(fc, 3, zeroP->GetOutShape());
	cnn.AddLayer(convolv);
	if (uselastRelu)
	{
		auto relu = std::make_shared<layers::mwReluLayer<Scalar>>(convolv->GetOutShape());
		cnn.AddLayer(relu);
	}
	return convolv;
}

template<class Scalar>
std::shared_ptr<layers::mwLayer<Scalar>> AddUpsampling(mwCNN<Scalar>& cnn, const size_t fc,
	const mwTensorView<Scalar>& inShape)
{
	auto up1 = std::make_shared<layers::mwUpsamplingLayer<Scalar>>(2, inShape);
	cnn.AddLayer(up1);
	auto zeroP = std::make_shared<layers::mwZeroPaddingLayer<Scalar>>(1, up1->GetOutShape());
	cnn.AddLayer(zeroP);
	auto convolv = std::make_shared<layers::mwConv2dLayer<Scalar>>(fc, 3, zeroP->GetOutShape());
	cnn.AddLayer(convolv);
	auto relu = std::make_shared<layers::mwReluLayer<Scalar>>(convolv->GetOutShape());
	cnn.AddLayer(relu);
	return convolv;
}

template<class Scalar>
std::shared_ptr<layers::mwLayer<Scalar>> AddPooling(
	mwCNN<Scalar>& cnn,
	const mwTensorView<Scalar>& inShape)
{
	auto maxP = std::make_shared<layers::mwMaxPoolLayer<Scalar>>(2, inShape);
	cnn.AddLayer(maxP);
	return maxP;
}

template<class Scalar>
void Create(const mwTensorView<Scalar>& inputShape, mwCNN<Scalar>& cnn)
{
	auto conv1 = AddConvolution(cnn, 32, inputShape); size_t c1Idx = cnn.Layers().size() - 1;
	auto pool1 = AddPooling(cnn, conv1->GetOutShape());
	auto conv2 = AddConvolution(cnn, 64, pool1->GetOutShape()); size_t c2Idx = cnn.Layers().size() - 1;
	auto pool2 = AddPooling(cnn, conv2->GetOutShape());
	auto conv3 = AddConvolution(cnn, 128, pool2->GetOutShape()); size_t c3Idx = cnn.Layers().size() - 1;
	auto pool3 = AddPooling(cnn, conv3->GetOutShape());
	auto conv4 = AddConvolution(cnn, 256, pool3->GetOutShape()); size_t c4Idx = cnn.Layers().size() - 1;
	cnn.AddLayer(std::make_shared<layers::mwDropOutLayer<Scalar>>(conv4->GetOutShape())); size_t d4 = cnn.Layers().size() - 1;
	auto pool4 = AddPooling(cnn, conv4->GetOutShape());
	auto conv5 = AddConvolution(cnn, 512, pool4->GetOutShape()); size_t c5Idx = cnn.Layers().size() - 1;
	cnn.AddLayer(std::make_shared<layers::mwDropOutLayer<Scalar>>(conv5->GetOutShape())); size_t d5 = cnn.Layers().size() - 1;
	auto up1 = AddUpsampling(cnn, 256, conv5->GetOutShape());
	cnn.AddLayer(std::make_shared<layers::mwConcatLayer<Scalar>>(
		cnn.Layers(), std::vector<size_t>(1, d4), up1->GetOutShape()));
	auto merge1 = cnn.Layers().back();
	auto conv6 = AddConvolution(cnn, 256, merge1->GetOutShape());
	auto up2 = AddUpsampling(cnn, 128, conv6->GetOutShape());
	cnn.AddLayer(std::make_shared<layers::mwConcatLayer<Scalar>>(
		cnn.Layers(), std::vector<size_t>(1, c3Idx), up2->GetOutShape()));
	auto conv7 = AddConvolution(cnn, 128, cnn.Layers().back()->GetOutShape());
	auto up3 = AddUpsampling(cnn, 64, conv7->GetOutShape());
	cnn.AddLayer(std::make_shared<layers::mwConcatLayer<Scalar>>(
		cnn.Layers(), std::vector<size_t>(1, c2Idx), up3->GetOutShape()));
	auto conv8 = AddConvolution(cnn, 64, cnn.Layers().back()->GetOutShape());
	auto up4 = AddUpsampling(cnn, 32, conv8->GetOutShape());
	cnn.AddLayer(std::make_shared<layers::mwConcatLayer<Scalar>>(
		cnn.Layers(), std::vector<size_t>(1, c1Idx), up4->GetOutShape()));

	auto conv9 = AddConvolution(cnn, 32, cnn.Layers().back()->GetOutShape());
	auto conv10 = AddConvolution(cnn, 32, cnn.Layers().back()->GetOutShape());
	auto conv11 = AddConvolution(cnn, 2, cnn.Layers().back()->GetOutShape());
	auto conv12 = AddConvolution(cnn, 1, cnn.Layers().back()->GetOutShape(), false);
	cnn.AddLayer(std::make_shared<layers::mwSigmoid<Scalar>>(conv12->GetOutShape()));
	cnn.Finalize();
}

template void Create<double>(const mwTensorView<double>&, mwCNN<double>&);
template void Create<float>(const mwTensorView<float>&, mwCNN<float>&);
}

