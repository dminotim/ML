#include "IO/include/dmDataReader.hpp"
#include <fstream>
#include <string>
#include <sstream>
#include "mwTensor.hpp"
#include "dmRGBImage.hpp"
#include "dmPainterIO.hpp"
#include <filesystem>
#include <utility>

namespace dmReader
{

std::vector<std::vector<double>> Read(const std::string& path)
{
	std::ifstream input(path);
	std::vector<std::vector<double>> res;
	for (std::string line; getline(input, line); )
	{
		std::stringstream ss;
		std::vector<double> temp;
		ss << line;
		double scalar;
		while (ss >> scalar)
		{
			temp.push_back(scalar);
		}
		res.push_back(temp);
	}
	return res;
}

template <class Scalar>
clusteriser::dmImage ConvertTensorToImg(const mwTensorView<Scalar>& tens)
{
	clusteriser::dmImage img(tens.ColCount(), tens.RowCount());
	for (size_t i = 0; i < tens.RowCount(); ++i)
	{
		for (size_t j = 0; j < tens.ColCount(); ++j)
		{
			img(j, i).red = tens(i, j, 0) * 255.;
			img(j, i).green = tens(i, j, 0) * 255.;
			img(j, i).blue = tens(i, j, 0) * 255.;
			img(j, i).alpha = 255.;
		}
	}
	return img;
}

template <class Scalar>
mwTensor<Scalar>
ConvertImgToTensor(const clusteriser::dmImage& img, const bool grayscale)
{
	if (grayscale)
	{
		mwTensor<Scalar> res(img.GetHeight(), img.GetWidth(), 1);
		for (size_t i = 0; i < img.GetWidth(); ++i)
		{
			for (size_t j = 0; j < img.GetHeight(); ++j)
			{
				res.ToView()(j, i, 0) = img(i, j).red /  255.;
			}
		}
		return res;
	}
	else
	{
		mwTensor<Scalar> res(img.GetHeight(), img.GetWidth(), 3);
		for (size_t i = 0; i < img.GetWidth(); ++i)
		{
			for (size_t j = 0; j < img.GetHeight(); ++j)
			{
				res.ToView()(j, i, 0) = img(i, j).red / 255.;
				res.ToView()(j, i, 1) = img(i, j).green / 255.;
				res.ToView()(j, i, 2) = img(i, j).blue / 255.;
			}
		}
		return res;
	}
}

template <class Scalar>
void DownloadXYImage(
	const std::string& trainXFolder,
	const std::string& trainYFolder,
	std::vector<mwTensor<Scalar>>& x,
	std::vector<mwTensor<Scalar>>& y)
{
	std::filesystem::directory_iterator end_itr;
	for (std::filesystem::directory_iterator itr(trainXFolder); itr != end_itr; ++itr)
	{
		clusteriser::dmImage img = clusteriser::IO::ReadImage(itr->path().string());
		x.push_back(ConvertImgToTensor<Scalar>(img));
	}
	for (std::filesystem::directory_iterator itr(trainYFolder); itr != end_itr; ++itr)
	{
		clusteriser::dmImage img = clusteriser::IO::ReadImage(itr->path().string());
		y.push_back(ConvertImgToTensor<Scalar>(img));
	}
}

template <class Scalar>
std::vector<std::pair<Scalar, clusteriser::dmImage >> DownloadMNIST(
	const std::string& testFilestFolder)
{
	std::vector<std::string> dirs;
	std::filesystem::directory_iterator end_itr;
	for (std::filesystem::directory_iterator itr(testFilestFolder); itr != end_itr; ++itr)
	{
		if (std::filesystem::is_directory(itr->path()))
		{
			dirs.push_back(itr->path().stem().string());
		}
	}

	std::vector<std::pair<Scalar, clusteriser::dmImage> > res;
	for (auto d : dirs)
	{
		int num = std::stoi(d);
		const std::string pathtoSub = testFilestFolder + d;
		size_t count = 0;
		for (std::filesystem::directory_iterator itr(pathtoSub);
			itr != end_itr && count < 800; ++itr)
		{
			clusteriser::dmImage img = clusteriser::IO::ReadImage(itr->path().string());
			res.emplace_back(Scalar(num), std::move(img));
			++count;
		}
	}
	return res;
}

template <class Scalar>
void ConvertMNISTToTensors(
	const std::vector<std::pair<Scalar, clusteriser::dmImage>>& data,
	std::vector<mwTensor<Scalar>>& x,
	std::vector<mwTensor<Scalar>>& y)
{
	x.clear();
	y.clear();
	for (auto& item : data)
	{
		mwTensor<Scalar> xt(item.second.GetHeight(), item.second.GetWidth(), 1);
		mwTensorView<Scalar> xView = xt.ToView();
		for (size_t i = 0; i < item.second.GetWidth(); ++i)
		{
			for (size_t j = 0; j < item.second.GetHeight(); ++j)
			{
				xView(j, i, 0) = Scalar(item.second(i, j).red) / Scalar(255);
			}
		}
		mwTensor<Scalar> yt(1, 1, 10);
		yt.ToView().ToVectorView()[int(item.first)] = 1.;
		x.push_back(xt);
		y.push_back(yt);
	}
}

template <class Scalar>
void ConvertToTensorView(const std::vector<mwTensor<Scalar>>& x,
	const std::vector<mwTensor<Scalar>>& y,
	std::vector<mwTensorView<Scalar>>& xv,
	std::vector<mwTensorView<Scalar>>& yv)
{
	xv.clear();
	yv.clear();
	for (size_t i = 0; i < x.size(); ++i)
	{
		xv.push_back(x[i].ToView());
		yv.push_back(y[i].ToView());
	}
}

template <class Scalar>
size_t GetMaxIndex(const mwTensorView<Scalar>& tens)
{
	auto vec = tens.ToVectorView();
	Scalar maxV = -10;
	size_t res = 0;
	for (int i = 0; i < vec.size(); ++i)
	{
		if (vec[i] > maxV)
		{
			maxV = vec[i];
			res = i;
		}
	}
	return res;
}

template void ConvertMNISTToTensors(
	const std::vector<std::pair<double, clusteriser::dmImage>>&,
	std::vector<mwTensor<double>>&,
	std::vector<mwTensor<double>>&);

template void ConvertMNISTToTensors(
	const std::vector<std::pair<float, clusteriser::dmImage>>&,
	std::vector<mwTensor<float>>&,
	std::vector<mwTensor<float>>&);

template clusteriser::dmImage ConvertTensorToImg(const mwTensorView<float>& tens);
template clusteriser::dmImage ConvertTensorToImg(const mwTensorView<double>& tens);

template void ConvertToTensorView(const std::vector<mwTensor<float>>& x,
	const std::vector<mwTensor<float>>& y,
	std::vector<mwTensorView<float>>& xv,
	std::vector<mwTensorView<float>>& yv);

template void ConvertToTensorView(const std::vector<mwTensor<double>>& x,
	const std::vector<mwTensor<double>>& y,
	std::vector<mwTensorView<double>>& xv,
	std::vector<mwTensorView<double>>& yv);

template size_t GetMaxIndex(const mwTensorView<double>& tens);
template size_t GetMaxIndex(const mwTensorView<float>& tens);

template std::vector<std::pair<float, clusteriser::dmImage >>
DownloadMNIST<float>(const std::string& testFilestFolder);

template std::vector<std::pair<double, clusteriser::dmImage >>
DownloadMNIST<double>(const std::string& testFilestFolder);


template void
DownloadXYImage<double>(
	const std::string& trainXFolder,
	const std::string& trainYFolder,
	std::vector<mwTensor<double>>& x,
	std::vector<mwTensor<double>>& y);

template void
DownloadXYImage<float>(
	const std::string& trainXFolder,
	const std::string& trainYFolder,
	std::vector<mwTensor<float>>& x,
	std::vector<mwTensor<float>>& y);

template 
mwTensor<float> ConvertImgToTensor(const clusteriser::dmImage& img,
	const bool grayscale);
template
mwTensor<double> ConvertImgToTensor(const clusteriser::dmImage& img,
	const bool grayscale);


}