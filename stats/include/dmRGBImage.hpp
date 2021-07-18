/******************************************************************************
(C) 2017 Author: Artem Avdoshkin
******************************************************************************/
#pragma once

#include <vector>
#include <cmath>

namespace clusteriser
{
typedef std::uint8_t uint8;
typedef std::uint64_t uint64;

struct Pixel
{
	Pixel() : red(0), green(0), blue(0), alpha(255) {}
	Pixel(const double value) : red(value), green(value), blue(value), alpha(255) {}
	Pixel(double r, double g, double b)
		: red(r), green(g), blue(b), alpha(255) {}

	Pixel(double r, double g, double b, double a)
		: red(r), green(g), blue(b), alpha(a) {}

	inline Pixel(const Pixel& tc)
	{
		red = tc.red;
		green = tc.green;
		blue = tc.blue;
		alpha = tc.alpha;
	};
	double red;
	double green;
	double blue;
	double alpha;

	inline const Pixel& operator=(const Pixel& tc)
	{
		if (this != &tc) {
			red = tc.red;
			green = tc.green;
			blue = tc.blue;
			alpha = tc.alpha;
		}
		return *this;
	};

	inline const Pixel& operator=(const double chanel)
	{
		red = chanel;
		green = chanel;
		blue = chanel;
		alpha = chanel;
		return *this;
	};


	inline const double& operator[](const int idx) const
	{
		return idx == 0 ? red : (idx == 1 ? green : blue);
	};

	inline double& operator[](const int idx)
	{
		return idx == 0 ? red : (idx == 1 ? green : blue);
	};

	inline bool operator==(const Pixel& tc)  const
	{
		return red == tc.red && green == tc.green
			&& blue == tc.blue && alpha == tc.alpha;
	};


	inline bool operator!=(const Pixel& tc)  const
	{
		return !((*this) == tc);
	};

	inline void operator+=(const Pixel& ta)
	{
		red += ta.red;
		green += ta.green;
		blue += ta.blue;
	};

	inline void operator/=(const Pixel& ta)
	{
		red /= ta.red;
		green /= ta.green;
		blue /= ta.blue;
	};

	inline void operator-=(const Pixel& ta)
	{
		red -= ta.red;
		green -= ta.green;
		blue -= ta.blue;
	};

	inline double operator~() const
	{
		return std::sqrt(
			red * red + green * green + blue * blue);
	};

	inline void operator*=(const double& scFactor)
	{
		red *= scFactor;
		green *= scFactor;
		blue *= scFactor;
	};

	inline void operator*=(const Pixel& scFactor)
	{
		red *= scFactor.red;
		green *= scFactor.green;
		blue *= scFactor.blue;
	};
	inline void operator/=(const double& scFactor)
	{
		red /= scFactor;
		green /= scFactor;
		blue /= scFactor;
	};
};

inline Pixel operator * (const Pixel& src, const double factor)
{
	Pixel res(src);
	res *= factor;
	return res;
};

inline Pixel operator * (const Pixel& first, const Pixel& second)
{
	Pixel res(first);
	res *= second;
	return res;
};


inline Pixel operator + (const Pixel& first, const Pixel& second)
{
	Pixel res(first);
	res += second;
	return res;
};

inline Pixel operator / (const Pixel& first, const Pixel& second)
{
	Pixel res(first);
	res /= second;
	return res;
};

inline Pixel operator - (const Pixel& first, const Pixel& second)
{
	Pixel res(first);
	res -= second;
	return res;
};


class dmImage
{
public:
	typedef std::vector<Pixel>	Pixels;
	dmImage() = default;
	dmImage(const dmImage&) = default;

	dmImage(const size_t width, const size_t height)
		: m_width(width), m_height(height), m_pixels(width * height, Pixel(0,0,0,0))
	{
	}

	dmImage(const size_t width, const size_t height, const Pixel& defColor)
		: m_width(width), m_height(height), m_pixels(width* height, defColor)
	{
	}

	dmImage(const std::vector<uint8>& imageData, const size_t width, const size_t height)
		: m_width(width), m_height(height)
	{
		m_pixels.reserve(m_width * m_height);
		for (size_t y = 0; y < m_height; ++y)
		{
			for (size_t x = 0; x < m_width; ++x) {
				const uint8 r = imageData[4 * y * m_width + 4 * x + 0]; //red
				const uint8 g = imageData[4 * y * m_width + 4 * x + 1]; //green
				const uint8 b = imageData[4 * y * m_width + 4 * x + 2]; //blue
				const uint8 a = imageData[4 * y * m_width + 4 * x + 3]; //alpha
				m_pixels.push_back(Pixel(r, g, b, a));
			}
		}
	}

	dmImage(const std::vector<Pixel>& pixels, const size_t width, const size_t height)
		: m_width(width), m_height(height), m_pixels(pixels)
	{
	}

	const size_t GetWidth() const
	{
		return m_width;
	}

	const size_t GetHeight() const
	{
		return m_height;
	}

	const Pixels& GetAllPixels() const
	{
		return m_pixels;
	}

	Pixels& GetAllPixels()
	{
		return m_pixels;
	}

	const Pixel& operator()(size_t w, size_t h) const
	{
		return m_pixels[h * m_width + w];
	}

	Pixel& operator()(size_t w, size_t h)
	{
		return m_pixels[h * m_width + w];
	}

private:
	Pixels m_pixels;
	size_t m_height;
	size_t m_width;
};

} // clusteriser