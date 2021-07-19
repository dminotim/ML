#pragma once
#include "mwTensor.hpp"
#include "mwLayer.hpp"
#include <memory>

namespace layers
{

template<typename Scalar>
struct mwConcatLayer : public mwLayer<Scalar>
{
	mwConcatLayer(const std::vector<std::shared_ptr<mwLayer<Scalar>>>& toConcat,
		const std::vector<size_t>& idxsToConcat,
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
	std::vector<std::shared_ptr<mwLayer<Scalar>>> Concatenated() const;
	const mwTensorView<Scalar> Input() const override;
	void SetDeltaToZero() override;

	template <class BinStream>
	void serialize(BinStream& stream)
	{
		stream << m_inputShape.RowCount()
			<< m_inputShape.ColCount()
			<< m_inputShape.Depth()
			<< m_idxs.size();
		for (size_t i = 0; i < m_idxs.size(); ++i)
		{
			stream << m_idxs[i];
		}
	}

	template <class BinStream>
	static std::shared_ptr<mwConcatLayer<Scalar> > deserialize(
		BinStream& stream,
		const std::vector<std::shared_ptr<mwLayer<Scalar>>>& allLayers)
	{
		size_t rowC, colC, depth;
		stream >> rowC >> colC >> depth;
		size_t concatIdxsSize;
		stream >> concatIdxsSize;
		std::vector<size_t> allIdxs(concatIdxsSize);
		for (size_t i = 0; i < concatIdxsSize; ++i)
		{
			stream >> allIdxs[i];
		}
		auto inShape = mwTensorView<Scalar>(
			nullptr, rowC, colC, depth);
		return std::make_shared<mwConcatLayer<Scalar>>(allLayers, allIdxs, inShape);
	}

private:
	mwTensorView<Scalar> m_in;
	mwTensor<Scalar> m_out;
	mwTensor<Scalar> m_delta;
	std::vector<std::shared_ptr<mwLayer<Scalar>>> m_concatenated;
	std::vector<size_t> m_idxs;
};

}