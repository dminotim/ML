#pragma once
#include <vector>

struct dmDataCollector
{
	dmDataCollector()
	{
	}

	void Collect(double y)
	{
		if (m_x.empty())
		{
			m_x.push_back(0);
		}
		else
		{
			m_x.push_back(m_x.back() + 1);
		}
		m_y.push_back(y);
	}
	std::vector<double> m_x;
	std::vector<double> m_y;
};
