#pragma once
#include "mwTensor.hpp"
#include "mwLayer.hpp"

namespace layers
{

template<typename Scalar>
struct mwUpsamplingLayer : public mwLayer<Scalar>
{
	mwUpsamplingLayer(const size_t kernel,
		const mwTensorView<Scalar>& inputShape);

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
	const mwTensorView<Scalar> Input() const override;
	void SetDeltaToZero() override;



	template <class BinStream>
	void serialize(BinStream& stream)
	{
		stream
			<< m_inputShape.RowCount()
			<< m_inputShape.ColCount()
			<< m_inputShape.Depth() << m_kernel;
	}

	template <class BinStream>
	static std::shared_ptr<mwUpsamplingLayer<Scalar> > deserialize(BinStream& stream)
	{
		size_t rowC, colC, depth, kernel;
		stream >> rowC >> colC >> depth >> kernel;
		auto inShape = mwTensorView<Scalar>(
			nullptr, rowC, colC, depth);
		return std::make_shared<mwUpsamplingLayer<Scalar>>(kernel, inShape);
	}
private:
	mwTensorView<Scalar> m_in;
	mwTensor<Scalar> m_out;
	mwTensor<Scalar> m_delta;
	size_t m_kernel;
};


}