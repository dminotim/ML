#pragma  once
#include "mwOptimizer.hpp"
#include <memory>
#include "mwLossFunction.hpp"
#include "mwLayer.hpp"

template<typename Scalar>
struct mwCNN
{
	mwCNN();
	void Fit(
		const std::vector<mwTensorView<Scalar>>& x,
		const std::vector<mwTensorView<Scalar>>& y,
		std::shared_ptr<mwOptimizer<Scalar>> opt,
		std::shared_ptr<mwLossFunction<Scalar>> loss,
		const size_t epochCount = 1,
		const size_t batchSize =  1);

	void AddLayer(const std::shared_ptr<layers::mwLayer<Scalar>> layer);

	mwTensorView<Scalar> Predict(const mwTensorView<Scalar>& x) const;

	std::vector<Scalar> GetAnaliticDeltas(
		std::shared_ptr<mwLossFunction<Scalar>> loss,
		const mwTensorView<Scalar>& x,
		const mwTensorView<Scalar>& y);

	Scalar GetNumericDeltas(
		std::shared_ptr<mwLossFunction<Scalar>> loss,
		const mwTensorView<Scalar>& x,
		const mwTensorView<Scalar>& y,
		size_t idx,
		Scalar h);

	std::vector<Scalar> GetNumericDeltas(
		std::shared_ptr<mwLossFunction<Scalar>> loss,
		const mwTensorView<Scalar>& x,
		const mwTensorView<Scalar>& y);

	mwTensor<Scalar> Cost(const mwTensorView<Scalar>& x,
		const mwTensorView<Scalar>& expectedY,
		std::shared_ptr<mwLossFunction<Scalar>> loss) const;
	void Finalize();
private:
	void SetZeros();
	void ValidateFinalization() const;
	Scalar TrainBatch(
		std::shared_ptr<mwOptimizer<Scalar>> opt,
		std::shared_ptr<mwLossFunction<Scalar>> loss,
		const std::vector<mwTensorView<Scalar>>& x,
		const std::vector<mwTensorView<Scalar>>& y,
		const size_t startIdx,
		const size_t endIdx);

	void Forward(const mwTensorView<Scalar>& x);
	void Backprop(const mwTensorView<Scalar>& delta);
	void CalcGrads(const mwTensorView<Scalar>& delta);
	bool IsLayerShouldBeSkippedInPrediction(std::shared_ptr<layers::mwLayer<Scalar>> layer) const;

	std::vector<std::shared_ptr<layers::mwLayer<Scalar>>> m_layers;
	std::vector<Scalar> m_weightsSpace;
	std::vector<Scalar> m_gradsSpace;
	bool m_isFinalized;
};


