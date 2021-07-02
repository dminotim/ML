#include "IO/include/dmDataReader.hpp"
#include <fstream>
#include <string>
#include <sstream>

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

}