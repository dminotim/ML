#pragma once

#include <string>
#include "dmMatrix.hpp"

namespace clusteriser
{

	dmImage ToGrayscale(const dmImage& image);


	
	std::vector<Pixel> ToGrayscale(const std::vector<Pixel>& source);


	std::vector<Pixel> RGBToLAB(const std::vector<Pixel>& source);

	std::vector<Pixel> LABToRGB(const std::vector<Pixel>& source);

	std::vector<clusteriser::Pixel> XYZToRGB(const std::vector<Pixel>& source);
	std::vector<clusteriser::Pixel> RGBToXYZ(const std::vector<Pixel>& source);

	double GetLabDist(const Pixel& first, const Pixel& second);

	std::vector<Pixel> ResizePixels(
		const std::vector<Pixel>& pixels,
		const size_t width,
		const size_t height,
		const size_t newWidth,
		const size_t newHeight);

	dmImage ResizeImage(const dmImage& image, const size_t width, const size_t height);

	std::vector<std::uint8_t> GetImageData(const dmImage& image);

	void CorrectPixelValues(std::vector<Pixel>& source);
}