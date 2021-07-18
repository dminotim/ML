#pragma once
#include <vector>
#include "mwTensor.hpp"
#include <string>
#include "dmRGBImage.hpp"

namespace dmReader
{
	std::vector<std::vector<double>> Read(const std::string& path);

	template <class Scalar>
	void ConvertMNISTToTensors(
		const std::vector<std::pair<Scalar, clusteriser::dmImage>>& data,
		std::vector<mwTensor<Scalar>>& x,
		std::vector<mwTensor<Scalar>>& y);

	template <class Scalar>
	clusteriser::dmImage ConvertTensorToImg(const mwTensorView<Scalar>& tens);

	template <class Scalar>
	void ConvertToTensorView(const std::vector<mwTensor<Scalar>>& x,
		const std::vector<mwTensor<Scalar>>& y,
		std::vector<mwTensorView<Scalar>>& xv,
		std::vector<mwTensorView<Scalar>>& yv);

	template <class Scalar>
	size_t GetMaxIndex(const mwTensorView<Scalar>& tens);

	template <class Scalar>
	std::vector<std::pair<Scalar, clusteriser::dmImage >> DownloadMNIST(const std::string& testFilestFolder);
}