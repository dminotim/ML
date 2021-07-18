#pragma once
#include "dmRGBImage.hpp"
#include <utility>
#include <vector>
#include <string>

namespace clusteriser
{

namespace IO
{

dmImage ReadImage(const std::string& src);

void WriteImage(const dmImage& image, const std::string& filename);

}

}