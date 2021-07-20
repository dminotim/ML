#pragma once
#include "mwTensor.hpp"
#include "mwLayer.hpp"
#include "mwHeInitialization.hpp"
#include <memory>

namespace layers
{

template<typename Scalar>
struct mwConv2dLayer : public mwLayer<Scalar>
{
	mwConv2dLayer(
		const size_t featuresCount,
		const size_t kernel,
		const mwTensorView<Scalar>& inputShape,
		const std::shared_ptr<mwInitialization<Scalar>> init
			= std::make_shared<mwHeInitialization<Scalar>>());

	mwTensorView<Scalar> GetOutShape() const override;

	mwLayerType GetType() const override;
	size_t OptimizedParamsCount() const override;
	void MapData(Scalar* weights, Scalar* gradient, Scalar* wokSpace) override;

	void Forward(const mwTensorView<Scalar>& input) override;
	void CalcGrads(const mwTensorView<Scalar>& nextDelta) override;
	void Backprop(const mwTensorView<Scalar>& nextDelta) override;
	const mwTensorView<Scalar> Output() const override;
	const mwTensorView<Scalar> GetDerivatives() const override;
	void Init() override;

	size_t Kernel() const;
	size_t FeaturesCount() const;
	const mwTensorView<Scalar> Input() const override;
	void SetDeltaToZero() override;

	template <class BinStream>
	void serialize(BinStream& stream)
	{
		stream 
			<< m_inputShape.RowCount()
			<< m_inputShape.ColCount()
			<< m_inputShape.Depth()
			<< m_kernel
			<< m_featuresCount;
	}

	template <class BinStream>
	static std::shared_ptr<mwConv2dLayer<Scalar> > deserialize(BinStream& stream)
	{
		size_t rowC, colC, depth, kernel, featuresCount;
		stream >> rowC >> colC >> depth >> kernel >> featuresCount;
		auto inShape = mwTensorView<Scalar>(
			nullptr, rowC, colC, depth);

		return std::make_shared<mwConv2dLayer<Scalar>>(
			featuresCount, kernel, inShape);
	}
private:
	mwTensorView<Scalar> m_in;
	mwTensor<Scalar> m_out;
	mwTensor<Scalar> m_delta;
	mwTensorView<Scalar> m_grads;
	mwTensorView<Scalar> m_weights;
	mwTensorView<Scalar> m_bias;
	mwTensorView<Scalar> m_biasGrads;
	mwTensorView<Scalar> m_workSpace;
	size_t m_kernel;
	size_t m_featuresCount;
	std::shared_ptr<mwInitialization<Scalar>> m_init;

};

}