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

	template <class BinStream>
	void serialize(BinStream& stream)
	{
		stream
			<< m_inputShape.RowCount()
			<< m_inputShape.ColCount()
			<< m_inputShape.Depth() << Padding();
	}

	template <class BinStream>
	static std::shared_ptr<mwZeroPaddingLayer<Scalar> > deserialize(BinStream& stream)
	{
		size_t rowC, colC, depth, padding;
		stream >> rowC >> colC >> depth >> padding;
		auto inShape = mwTensorView<Scalar>(
			nullptr, rowC, colC, depth);
		return std::make_shared<mwZeroPaddingLayer<Scalar>>(padding, inShape);
	}
private:
	mwTensorView<Scalar> m_in;
	mwTensor<Scalar> m_out;
	mwTensor<Scalar> m_delta;
	size_t m_padding;
};

}