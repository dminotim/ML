#pragma once
#include "mwZeroPaddingLayer.hpp"
#include "mwUpsamplingLayer.hpp"
#include "mwSoftMax.hpp"
#include "mwReluLayer.hpp"
#include "mwMaxPoolLayer.hpp"
#include "mwFCLayer.hpp"
#include "mwDropOutLayer.hpp"
#include "mwConv2dLayer.hpp"
#include "mwConcatLayer.hpp"
#include "mwSigmoid.hpp"


namespace serializer
{
template<class Scalar,class BinStream>
void SerializeLayer(BinStream& stream,
	const layers::mwLayerType type,
	layers::mwLayer<Scalar>& layer,
	const std::vector<std::shared_ptr<layers::mwLayer<Scalar>>>& /*allLayers*/)
{
	switch (type)
	{
	case layers::mwLayerType::CONCAT:
		static_cast<layers::mwConcatLayer<Scalar>&>(layer).serialize(stream);
		break;
	case layers::mwLayerType::CONVOLUTION:
		static_cast<layers::mwConv2dLayer<Scalar>&>(layer).serialize(stream);
		break;
	case layers::mwLayerType::DROP_OUT:
		static_cast<layers::mwDropOutLayer<Scalar>&>(layer).serialize(stream);
		break;
	case layers::mwLayerType::FULLY_CONNECTED:
		static_cast<layers::mwFCLayer<Scalar>&>(layer).serialize(stream);
		break;
	case layers::mwLayerType::MAX_POOL:
		static_cast<layers::mwMaxPoolLayer<Scalar>&>(layer).serialize(stream);
		break;
	case layers::mwLayerType::RELU:
		static_cast<layers::mwReluLayer<Scalar>&>(layer).serialize(stream);
		break;
	case layers::mwLayerType::SOFTMAX:
		static_cast<layers::mwSoftMax<Scalar>&>(layer).serialize(stream);
		break;
	case layers::mwLayerType::UPSAMPLING:
		static_cast<layers::mwUpsamplingLayer<Scalar>&>(layer).serialize(stream);
		break;
	case layers::mwLayerType::ZERO_PADDING:
		static_cast<layers::mwZeroPaddingLayer<Scalar>&>(layer).serialize(stream);
		break;
	case layers::mwLayerType::SIGMOID:
		static_cast<layers::mwSigmoid<Scalar>&>(layer).serialize(stream);
		break;
	default:
		break;
	}
}

template<class Scalar, class BinStream>
std::shared_ptr<layers::mwLayer<Scalar>>
DeserializeLayer(BinStream& stream,
	const layers::mwLayerType type,
	const std::vector<std::shared_ptr<layers::mwLayer<Scalar>>>& allLayers)
{
	switch (type)
	{
	case layers::mwLayerType::CONCAT:
		return layers::mwConcatLayer<Scalar>::deserialize(stream, allLayers);
	case layers::mwLayerType::CONVOLUTION:
		return layers::mwConv2dLayer<Scalar>::deserialize(stream);
	case layers::mwLayerType::DROP_OUT:
		return layers::mwDropOutLayer<Scalar>::deserialize(stream);
	case layers::mwLayerType::FULLY_CONNECTED:
		return layers::mwFCLayer<Scalar>::deserialize(stream);
	case layers::mwLayerType::MAX_POOL:
		return layers::mwMaxPoolLayer<Scalar>::deserialize(stream);
	case layers::mwLayerType::RELU:
		return layers::mwReluLayer<Scalar>::deserialize(stream);
	case layers::mwLayerType::SOFTMAX:
		return layers::mwSoftMax<Scalar>::deserialize(stream);
	case layers::mwLayerType::UPSAMPLING:
		return layers::mwUpsamplingLayer<Scalar>::deserialize(stream);
	case layers::mwLayerType::ZERO_PADDING:
		return layers::mwZeroPaddingLayer<Scalar>::deserialize(stream);
	case layers::mwLayerType::SIGMOID:
		return layers::mwSigmoid<Scalar>::deserialize(stream);
	default:
		return nullptr;
	}
}

}