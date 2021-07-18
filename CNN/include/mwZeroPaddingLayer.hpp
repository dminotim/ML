#pragma once
#include "mwTensor.hpp"
#include "mwLayer.hpp"

namespace layers
{

template<typename Scalar>
struct mwZeroPaddingLayer : public mwLayer<Scalar>
{
	mwZeroPaddingLayer(
		const size_t padidngSize,
		const mwTensorView<Scalar>& inputShape);

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
	size_t Padding() const;

	const mwTensorView<Scalar> Input() const override;
	void SetDeltaToZero() override;
private:
	mwTensorView<Scalar> m_in;
	mwTensor<Scalar> m_out;
	mwTensor<Scalar> m_delta;
	size_t m_padding;
};

}