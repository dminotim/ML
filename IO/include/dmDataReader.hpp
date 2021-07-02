#pragma once
#include <vector>
#include "dmMatrix.hpp"
#include <string>

namespace dmReader
{
	std::vector<std::vector<double>> Read(const std::string& path);
}