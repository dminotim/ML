//#include "dmRGBImage.hpp"
//#include ".dmImageUtils.hpp"

#include <string>
#include <cmath>
#include "lodepng.h"
#include "dmRGBImage.hpp"
#include <algorithm>
#include "dmImageUtils.hpp"

namespace clusteriser
{

std::vector<uint8> GetImageData(const dmImage& image)
{
	std::vector<uint8> imageData(image.GetWidth() * image.GetHeight() * 4);
	for (size_t y = 0; y < image.GetHeight(); ++y)
	{
		for (size_t x = 0; x < image.GetWidth(); ++x) {
			imageData[4 * y * image.GetWidth() + 4 * x + 0] = uint8(image(x, y).red);
			imageData[4 * y * image.GetWidth() + 4 * x + 1] = uint8(image(x, y).green);
			imageData[4 * y * image.GetWidth() + 4 * x + 2] = uint8(image(x, y).blue);
			imageData[4 * y * image.GetWidth() + 4 * x + 3] = uint8(image(x, y).alpha);;
		}
	}
	return imageData;
}


void CorrectPixelValues(std::vector<Pixel>& source)
{
	for (auto& pixel : source)
	{
		for (size_t i = 0; i < 3; ++i)
		{
			if (std::isnan(pixel[i]))
			{
				pixel[i] = 0;
			}
			if (pixel[i] > 254)
			{
				pixel[i] = 255.;
			}
			if (pixel[i] < 1)
			{
				pixel[i] = 0;
			}

		}

		
	}
}

template<class T>
inline T Clamp(const T value, const T low, const T high)
{
	if (value < low)
		return low;

	if (value > high)
		return high;

	return value;
}


dmImage ToGrayscale(const dmImage& image)
{
	dmImage res = image;
	for (size_t i = 0; i < res.GetWidth(); ++i)
	{
		for (size_t j = 0; j < res.GetHeight(); ++j)
		{
			Pixel& pixel = res(i, j);
			double brightness = uint8(
				0.2126 * double(pixel.red) + 0.7152 * double(pixel.green) + 0.0722 * double(pixel.blue)) ;
			pixel.red = pixel.green = pixel.blue = brightness;
		}
	}
	return res;
}

//clusteriser::Pixel ApplyMatrixToPixel(
//	const dmImage& image,
//	const int x,
//	const int y,
//	const dmMatrix<double, 3, 3> & matrix)
//{
//	return (image(x, y) * matrix[0][0])
//		+ (image(x + 1, y) * matrix[0][1])
//		+ (image(x + 2, y) * matrix[0][2])
//		+ (image(x, y + 1) * matrix[1][0])
//		+ (image(x + 1, y + 1) * matrix[1][1])
//		+ (image(x + 2, y + 1) * matrix[1][2])
//		+ (image(x, y + 2) * matrix[2][0])
//		+ (image(x + 1, y + 2) * matrix[2][1])
//		+ (image(x + 2, y + 2) * matrix[2][2]);
//}
//
//double ApplyMatrixToRedCh(
//	const dmImage& image,
//	const int x,
//	const int y,
//	const dmMatrix<double, 3, 3> & matrix)
//{
//	return (image(x - 1, y - 1).red * matrix[0][0])
//		+ (image(x, y - 1).red * matrix[0][1])
//		+ (image(x + 1, y - 1).red * matrix[0][2])
//		+ (image(x - 1, y).red * matrix[1][0])
//		+ (image(x, y).red * matrix[1][1])
//		+ (image(x + 1, y).red * matrix[1][2])
//		+ (image(x - 1, y + 1).red * matrix[2][0])
//		+ (image(x, y + 1).red * matrix[2][1])
//		+ (image(x + 1, y + 1).red * matrix[2][2]);
//}
//
//double ApplyMatrixToGreenCh(
//	const dmImage& image,
//	const int x,
//	const int y,
//	const dmMatrix<double, 3, 3> & matrix)
//{
//	return (image(x, y).green * matrix[0][0])
//		+ (image(x + 1, y).green * matrix[0][1])
//		+ (image(x + 2, y).green * matrix[0][2])
//		+ (image(x, y + 1).green * matrix[1][0])
//		+ (image(x + 1, y + 1).green * matrix[1][1])
//		+ (image(x + 2, y + 1).green * matrix[1][2])
//		+ (image(x, y + 2).green * matrix[2][0])
//		+ (image(x + 1, y + 2).green * matrix[2][1])
//		+ (image(x + 2, y + 2).green * matrix[2][2]);
//}
//
//double ApplyMatrixToBlueCh(
//	const dmImage& image,
//	const int x,
//	const int y,
//	const dmMatrix<double, 3, 3> & matrix)
//{
//	return (image(x, y).blue * matrix[0][0])
//		+ (image(x + 1, y).blue * matrix[0][1])
//		+ (image(x + 2, y).blue * matrix[0][2])
//		+ (image(x, y + 1).blue * matrix[1][0])
//		+ (image(x + 1, y + 1).blue * matrix[1][1])
//		+ (image(x + 2, y + 1).blue * matrix[1][2])
//		+ (image(x, y + 2).blue * matrix[2][0])
//		+ (image(x + 1, y + 2).blue * matrix[2][1])
//		+ (image(x + 2, y + 2).blue * matrix[2][2]);
//}

Pixel RGBToLAB(const Pixel& source)
{
	double r = source.red / 255;
	double g = source.green / 255;
	double b = source.blue / 255;

	r = (r > 0.04045) ? std::pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
	g = (g > 0.04045) ? std::pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
	b = (b > 0.04045) ? std::pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

	double x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047;
	double y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000;
	double z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;

	x = (x > 0.008856) ? std::pow(x, 1. / 3.) : (7.787 * x) + 16. / 116.;
	y = (y > 0.008856) ? std::pow(y, 1. / 3.) : (7.787 * y) + 16. / 116.;
	z = (z > 0.008856) ? std::pow(z, 1. / 3.) : (7.787 * z) + 16. / 116.;

	return Pixel((116. * y) - 16., 500. * (x - y), 200. * (y - z));
}

Pixel RGBToXYZ(const Pixel& source)
{
	double r = source.red / 255;
	double g = source.green / 255;
	double b = source.blue / 255;

	r = (r > 0.04045) ? std::pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
	g = (g > 0.04045) ? std::pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
	b = (b > 0.04045) ? std::pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

	double x = (r * 0.4124 + g * 0.3576 + b * 0.1805);
	double y = (r * 0.2126 + g * 0.7152 + b * 0.0722);
	double z = (r * 0.0193 + g * 0.1192 + b * 0.9505);

	return Pixel(x, y, z);
}

std::vector<clusteriser::Pixel> RGBToXYZ(const std::vector<Pixel>& source)
{
	std::vector<clusteriser::Pixel> res(source.size());
	for (size_t i = 0; i < source.size(); ++i)
	{
		res[i] = RGBToXYZ(source[i]);
	}
	return res;
}

Pixel XYZToRGB(const Pixel& source)
{
	double x = source.red;
	double y = source.green;
	double z = source.blue;

	double r = x * 3.2406 + y * -1.5372 + z * -0.4986;
	double g = x * -0.9689 + y * 1.8758 + z * 0.0415;
	double b = x * 0.0557 + y * -0.2040 + z * 1.0570;

	r = (r > 0.0031308) ? (1.055 * std::pow(r, 1. / 2.4) - 0.055) : 12.92 * r;
	g = (g > 0.0031308) ? (1.055 * std::pow(g, 1. / 2.4) - 0.055) : 12.92 * g;
	b = (b > 0.0031308) ? (1.055 * std::pow(b, 1. / 2.4) - 0.055) : 12.92 * b;

	return Pixel(std::max(0., std::min(1., r)) * 255,
		std::max(0., std::min(1., g)) * 255,
		std::max(0., std::min(1., b)) * 255);
}

std::vector<clusteriser::Pixel> XYZToRGB(const std::vector<Pixel>& source)
{
	std::vector<clusteriser::Pixel> res(source.size());
	for (size_t i = 0; i < source.size(); ++i)
	{
		res[i] = XYZToRGB(source[i]);
	}
	return res;
}

Pixel LABToRGB(const Pixel& source)
{
	double y = (source.red + 16) / 116;
	double x = source.green / 500 + y;
	double z = y - source.blue / 200;


	x = 0.95047 * ((x * x * x > 0.008856) ? x * x * x : (x - 16. / 116.) / 7.787);
	y = 1.00000 * ((y * y * y > 0.008856) ? y * y * y : (y - 16. / 116.) / 7.787);
	z = 1.08883 * ((z * z * z > 0.008856) ? z * z * z : (z - 16. / 116.) / 7.787);

	double r = x * 3.2406 + y * -1.5372 + z * -0.4986;
	double g = x * -0.9689 + y * 1.8758 + z * 0.0415;
	double b = x * 0.0557 + y * -0.2040 + z * 1.0570;

	r = (r > 0.0031308) ? (1.055 * std::pow(r, 1. / 2.4) - 0.055) : 12.92 * r;
	g = (g > 0.0031308) ? (1.055 * std::pow(g, 1. / 2.4) - 0.055) : 12.92 * g;
	b = (b > 0.0031308) ? (1.055 * std::pow(b, 1. / 2.4) - 0.055) : 12.92 * b;

	return Pixel(std::max(0., std::min(1., r)) * 255,
		std::max(0., std::min(1., g)) * 255,
		std::max(0., std::min(1., b)) * 255);
}

std::vector<clusteriser::Pixel> RGBToLAB(const std::vector<Pixel>& source)
{
	std::vector<clusteriser::Pixel> res(source.size());
	for (size_t i = 0; i < source.size(); ++i)
	{
		res[i] = RGBToLAB(source[i]);
	}
	return res;
}

std::vector<clusteriser::Pixel> LABToRGB(const std::vector<Pixel>& source)
{
	std::vector<clusteriser::Pixel> res(source.size());
	for (size_t i = 0; i < source.size(); ++i)
	{
		res[i] = LABToRGB(source[i]);
	}
	return res;
}

std::vector<Pixel> ResizePixels(
	const std::vector<Pixel>& pixels,
	const size_t width,
	const size_t height,
	const size_t newWidth,
	const size_t newHeight)
{
	std::vector<Pixel> res(newWidth * newHeight);
	int xRatio = (int)((width << 16) / newWidth) + 1;
	int yRatio = (int)((height << 16) / newHeight) + 1;

	int x2, y2;
	for (int i = 0; i < newHeight; ++i) {
		for (int j = 0; j < newWidth; ++j) {
			x2 = ((j * xRatio) >> 16);
			y2 = ((i * yRatio) >> 16);
			res[(i * newWidth) + j] = pixels[(y2 * width) + x2];
		}
	}
	return res;
}


clusteriser::dmImage ResizeImage(const dmImage& image, const size_t width, const size_t height)
{
	return dmImage(
		ResizePixels(image.GetAllPixels(),
			image.GetWidth(), image.GetHeight(), width, height),
		width, height);
}

std::vector<Pixel> ToGrayscale(const std::vector<Pixel>& source)
{
	std::vector<Pixel> res(source.size());

	for (size_t i = 0; i < source.size(); ++i)
	{
		Pixel pixel = source[i];
		double brightness = uint8(
			0.2126 * double(pixel.red) + 0.7152 * double(pixel.green) + 0.0722 * double(pixel.blue));
		pixel.red = pixel.green = pixel.blue;
		res[i] = pixel;
	}
	return res;
}

}
