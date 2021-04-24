// MatrixApp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "dmMatrix.hpp"
#include <vector>
#include <algorithm>
#include <stdlib.h>

int main()
{
	dmMatrix<int> A(16, 3200);
	dmMatrix<int> B(3200, 16);
	std::vector<int> factors = { 2, 2,2,2,2 };
	int id = 0;
	for (int i = 0; i < A.GetRowCount(); ++i)
	{
		for (int j = 0; j < A.GetColCount(); ++j)
		{
			A(i, j) = rand() % 10;
		}
	}

	for (int i = 0; i < B.GetRowCount(); ++i)
	{
		for (int j = 0; j < B.GetColCount(); ++j)
		{
			B(i, j) = rand() % 10;
		}
	}
	/*auto ans1 = A.Multiply(B.Multiply(factors));
	auto ans2 = A.Multiply(B).Multiply(factors);*/
	A.Multiply(B);
	auto multRes = A.MultiplySlow(B).m_values;
	std::sort(multRes.begin(), multRes.end());
	std::cout << std::endl;
	for (auto& el : multRes)
	{
		std::cout << el << " ";
	}
	std::cout << std::endl;

	/*for (int i = 0; i < ans1.size(); ++i)
	{
		std::cout << ans1[i] << " ";
	}
	std::cout << std::endl;
	for (int i = 0; i < ans1.size(); ++i)
	{
		std::cout << ans2[i] << " ";
	}*/
	std::cout << "Hello World!\n";
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
