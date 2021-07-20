#pragma  once
#include "mwTensor.hpp"
#include "mwCNNUtils.hpp"
#include <iostream>

namespace mwCNNUtils
{
template <class Scalar>
void ToColumnImage(const mwTensorView<Scalar>& src,
	const size_t kernel,
	const size_t padding,
	mwTensorView<Scalar>& dst)
{
	dmMatrixView<Scalar> outM = dst(0);
	const size_t step = kernel - 1;
	size_t rowIdx = 0;
	for (size_t d = 0; d < src.Depth(); ++d)
	{
		dmMatrixView<Scalar> m = src(d);
		size_t kernelSize = kernel * kernel;
		
		for (int ki = 0; ki < kernel; ++ki)
		{
			for (int kj = 0; kj < kernel; ++kj)
			{
				size_t colIdx = 0;
				for (int i = 0; i < m.RowCount(); ++i)
				{
					bool isValidI = (i + (step - ki) -int(padding) < m.RowCount())
						&& (int(i - ki) + int(padding) >= 0);
					for (int j = 0; isValidI && j < m.ColCount(); ++j)
					{
						bool isValidJ = (j + (step - kj) - int(padding) < m.ColCount())
							&& (int(j - kj) + int(padding) >= 0);
						if(!isValidJ)
							continue;
						outM(rowIdx, colIdx) = m(i, j);
						++colIdx;
					}
				}
				++rowIdx;
			}
		}
	}

}

template void ToColumnImage<double>(const mwTensorView<double>&, const size_t, const size_t, mwTensorView<double>&);
template void ToColumnImage<float>(const mwTensorView<float>&, const size_t, const size_t, mwTensorView<float>&);
}

