#pragma once
#include "mwTensor.hpp"
#include "mwLayer.hpp"
#include "mwInitialization.hpp"
#include "mwHeInitialization.hpp"
#include <memory>

namespace layers
{

template<typename Scalar>
struct mwFCLayer : public mwLayer<Scalar>
{
	mwFCLayer(
		const size_t outSize,
		const mwTensorView<Scalar>& inputShape,
		const std::shared_ptr<mwInitialization<Scalar>> init
			= std::make_shared<mwHeInitialization<Scalar>>());

	mwTensorView<Scalar> GetOutShape() const override;

	mwLayerType GetType() const override;
	size_t OptimizedParamsCount() const override;
	void MapData(Scalar* weights, Scalar* gradient) override;

	void Forward(const mwTensorView<Scalar>& input) override;
	void CalcGrads(const mwTensorView<Scalar>& nextDelta) override;
	void Backprop(const mwTensorView<Scalar>& nextDelta) override;
	const mwTensorView<Scalar> Output() const override;
	const mwTensorView<Scalar> GetDerivatives() const override;
	void Init() override;

	const mwTensorView<Scalar> Input() const override;
	void SetDeltaToZero() override;

	template <class BinStream>
	void serialize(BinStream& stream)
	{
		stream
			<< m_inputShape.RowCount()
			<< m_inputShape.ColCount()
			<< m_inputShape.Depth() << m_out.ToView().Size();
	}

	template <class BinStream>
	static std::shared_ptr<mwFCLayer<Scalar> > deserialize(BinStream& stream)
	{
		size_t rowC, colC, depth, outSize;
		stream >> rowC >> colC >> depth >> outSize;
		auto inShape = mwTensorView<Scalar>(nullptr, rowC, colC, depth);
		return std::make_shared<mwFCLayer<Scalar>>(outSize, inShape);
	}
private:
	mwTensorView<Scalar> m_in;
	mwTensor<Scalar> m_out;
	mwTensor<Scalar> m_delta;
	mwTensorView<Scalar> m_grads;
	mwTensorView<Scalar> m_weights;
	mwTensorView<Scalar> m_bias;
	mwTensorView<Scalar> m_biasGrads;
	std::shared_ptr<mwInitialization<Scalar>> m_init;
};

}