#pragma once
#include <vector>
#include "dmMatrix.hpp"

namespace dmStatsCore
{
	template<typename T>
	struct has_const_iterator
	{
	private:
		typedef char                      yes;
		typedef struct { char array[2]; } no;

		template<typename C> static yes test(typename C::const_iterator*);
		template<typename C> static no  test(...);
	public:
		static const bool value = sizeof(test<T>(0)) == sizeof(yes);
		typedef T type;
	};

	template <typename T>
	struct has_begin_end
	{
		template<typename C> static char(&f(typename std::enable_if<
			std::is_same<decltype(static_cast<typename C::const_iterator(C::*)() const>(&C::begin)),
			typename C::const_iterator(C::*)() const>::value, void>::type*))[1];

		template<typename C> static char(&f(...))[2];

		template<typename C> static char(&g(typename std::enable_if<
			std::is_same<decltype(static_cast<typename C::const_iterator(C::*)() const>(&C::end)),
			typename C::const_iterator(C::*)() const>::value, void>::type*))[1];

		template<typename C> static char(&g(...))[2];

		static bool const beg_value = sizeof(f<T>(0)) == 1;
		static bool const end_value = sizeof(g<T>(0)) == 1;
	};

	template<typename T>
	struct is_container :
		std::integral_constant<bool, has_const_iterator<T>::value&& has_begin_end<T>::beg_value&& has_begin_end<T>::end_value>
	{ };
	template <class Container>
	typename Container::value_type GetMean(const Container& src)
	{
		static_assert(is_container<Container>::value,
			"GetMean expect container type as input");
		typedef typename Container::value_type T;
		if (src.empty())
			throw std::exception("Impossible to calculate mean for empty container");
		T sum = *std::begin(src);
		for (auto it = std::next(std::begin(src)); it != std::end(src); ++it)
		{
			sum += *it;
		}
		return sum / src.size();
	}

	template <class Container>
	typename Container::value_type GetVariance(const Container& src)
	{
		static_assert(is_container<Container>::value,
			"GetMean expect container type as input");
		typedef typename Container::value_type T;
		if (src.empty())
			throw std::exception("Impossible to calculate variance for empty container");
		const T meanV = GetMean(src);
		T sum = (meanV - *std::begin(src)) * (meanV - *std::begin(src));
		for (auto it = std::next(std::begin(src)); it != std::end(src); ++it)
		{
			sum += (*it - meanV) * (*it - meanV);
		}
		return sum / src.size();
	}

	template <class Container>
	typename Container::value_type GetStandartDeviation(const Container& src)
	{
		return std::sqrt(GetVariance(src));
	}

	template<class Container>
	dmMatrix<typename Container::value_type> GetCovariationMatrix(
		const std::vector<Container>& samples)
	{
		if (samples.empty())
			throw std::exception("Samples vector should not be empty");
		dmMatrix<typename Container::value_type> res(samples[0].size(), samples[0].size());
		std::vector<typename Container::value_type> mean(samples[0].size(), 0);
		for (const Container & c : samples)
		{
			for (size_t i = 0; i < c.size(); ++i)
			{
				mean[i] += c[i];
			}
		}
		typedef typename Container::value_type T;
		for (size_t i = 0; i < mean.size(); ++i)
		{
			mean[i] /= T(samples.size());
		}
		for (size_t i = 0; i < res.GetRowCount(); ++i)
		{
			for (size_t j = 0; j < res.GetColCount(); ++j)
			{
				for (size_t k = 0; k < samples.size(); ++k)
				{
					res(i, j) += (samples[k][i] - mean[i]) * (samples[k][j] - mean[j]);
				}
				res(i, j) /= T(samples.size() - 1);
			}
		}
		return res;
	}

	template<class Container>
	dmMatrix<typename Container::value_type> GetCorrelationMatrix(
		const std::vector<Container>& samples)
	{
		auto matr = GetCovariationMatrix(samples);
		dmMatrix<typename Container::value_type> res(matr.GetRowCount(), matr.GetColCount());
		for (size_t i = 0; i < matr.GetRowCount(); ++i)
		{
			for (size_t j = 0; j < matr.GetColCount(); ++j)
			{
				res(i, j) = matr(i, j) / (std::sqrt(matr(i, i)) * std::sqrt(matr(j, j)));
			}
		}
		return res;
	}
	struct dmPolynom
	{
		dmPolynom(const std::vector<double>& coeff)
			: m_coeff(coeff)
		{}

		double Evaluate(const double x) const
		{
			double powX = x;
			double res = m_coeff[0];
			for (size_t i = 1; i < m_coeff.size(); ++i)
			{
				res += powX * m_coeff[i];
				powX *= x;
			}
			return res;
		}

		const std::vector<double> m_coeff;
	};

	dmPolynom PolyFit(const std::vector<double>& x,
		const std::vector<double>& y, const double degree)
	{
		dmMatrix<double> m(x.size(), degree + 1);
		for (size_t i = 0; i < x.size(); ++i)
		{
			for (size_t j = 0; j <= degree; ++j)
			{
				m(i, j) = std::pow(x[i], double(j));
			}
		}
		dmMatrix<double> mt(m);
		mt.Transpose();
		dmMatrix<double> mmt = mt * m;
		std::vector<double> yt = mt * y;
		return dmLDLDecompose::Solve(mmt, yt);
	}
	 
	template<class Container>
	dmMatrix<typename Container::value_type> GetR2Matrix(
		const std::vector<Container>& samples)
	{
		typedef typename Container::value_type T;

		dmMatrix<T> res(samples[0].size(), samples[0].size());
		std::vector<std::vector<T>> columnSamples(samples[0].size());
		for (const Container& c : samples)
		{
			for (size_t i = 0; i < c.size(); ++i)
			{
				columnSamples[i].push_back(c[i]);
			}
		}
	
		auto covar = GetCovariationMatrix(samples);
		for (size_t i = 0; i < res.GetRowCount(); ++i)
		{
			for (size_t j = 0; j < res.GetColCount(); ++j)
			{
				auto polynom = PolyFit(columnSamples[i], columnSamples[j], 1);
				double varianceLine = 0;
				for (size_t k = 0; k < columnSamples[i].size(); ++k)
				{
					const double y = polynom.Evaluate(columnSamples[i][k]);
					varianceLine += (y - columnSamples[j][k])
						* (y - columnSamples[j][k]);
				}
				varianceLine /= double(columnSamples[i].size() - 1);
				res(i, j) = (covar(j, j) - varianceLine) / covar(j, j);
			}
		}
		return res;
	}

}