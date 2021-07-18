#include "dmPainterIO.hpp"
#include "dmRGBImage.hpp"
#include "lodepng.h"

#include <utility>
#include <vector>

namespace clusteriser
{

namespace IO
{

dmImage ReadImage(const std::string& filename)
{
	unsigned width, height;
	std::vector<uint8> imageData;
	unsigned error = lodepng::decode(imageData, width, height, filename);
	if (error != 0) {
		throw std::exception(lodepng_error_text(error));
	}
	return dmImage(imageData, width, height);
}

void WriteImage(const dmImage& image, const std::string& filename)
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
	lodepng::encode(filename, imageData, image.GetWidth(), image.GetHeight());
}


}

}