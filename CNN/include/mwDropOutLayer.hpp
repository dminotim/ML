#pragma once
#include "mwTensor.hpp"
#include "mwLayer.hpp"
#include <memory>

namespace layers
{

template<typename Scalar>
struct mwDropOutLayer : public mwLayer<Scalar>
{
	mwDropOutLayer(
		const mwTensorView<Scalar>& inputShape, const Scalar treshold = Scalar(0.5), const int seed = 3731);

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
	Scalar Treshold() const;
	const mwTensorView<Scalar> Input() const override;
	void SetDeltaToZero() override;

	template <class BinStream>
	void serialize(BinStream& stream)
	{
		stream << m_inputShape.RowCount()
			<< m_inputShape.ColCount()
			<< m_inputShape.Depth()
			<< Treshold();
	}

	template <class BinStream>
	static std::shared_ptr<mwDropOutLayer<Scalar> > deserialize(BinStream& stream)
	{
		size_t rowC, colC, depth;
		Scalar treshold;
		stream >> rowC >> colC >> depth >> treshold;
		auto inShape = mwTensorView<Scalar>(
			nullptr, rowC, colC, depth);

		return std::make_shared<mwDropOutLayer<Scalar>>(inShape, treshold);
	}
private:
	mwTensorView<Scalar> m_in;
	mwTensor<Scalar> m_out;
	mwTensor<Scalar> m_delta;
	mwTensor<int> m_activated;
	Scalar m_treshold;
	std::default_random_engine m_engine;
	std::uniform_real_distribution<Scalar> m_uniformDist;
};

}