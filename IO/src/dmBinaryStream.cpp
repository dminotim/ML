#include "dmBinaryStream.hpp"
#include <limits>
#include <cmath>

template <>
void dmBinOStream::save(const unsigned long long& t) {
	save((unsigned char)((t) & 0xFF));
	save((unsigned char)((t >> 8) & 0xFF));
	save((unsigned char)((t >> 16) & 0xFF));
	save((unsigned char)((t >> 24) & 0xFF));
	save((unsigned char)((t >> 32) & 0xFF));
	save((unsigned char)((t >> 40) & 0xFF));
	save((unsigned char)((t >> 48) & 0xFF));
	save((unsigned char)((t >> 56) & 0xFF));
};

template <>
void dmBinIStream::load(unsigned long long& t) {
	unsigned char buf;
	load(buf); t = static_cast<unsigned long long>(buf);
	load(buf); t += static_cast<unsigned long long>(buf) << 8;
	load(buf); t += static_cast<unsigned long long>(buf) << 16;
	load(buf); t += static_cast<unsigned long long>(buf) << 24;
	load(buf); t += static_cast<unsigned long long>(buf) << 32;
	load(buf); t += static_cast<unsigned long long>(buf) << 40;
	load(buf); t += static_cast<unsigned long long>(buf) << 48;
	load(buf); t += static_cast<unsigned long long>(buf) << 56;
}

template <>
void dmBinOStream::save(const unsigned int& t) {
	save((unsigned char)((t) & 0xFF));
	save((unsigned char)((t >> 8) & 0xFF));
	save((unsigned char)((t >> 16) & 0xFF));
	save((unsigned char)((t >> 24) & 0xFF));
};
template <>
void dmBinIStream::load(unsigned int& t) {
	unsigned char buf;
	load(buf); t = buf;
	load(buf); t += buf << 8;
	load(buf); t += buf << 16;
	load(buf); t += buf << 24;
}
template <>
void dmBinOStream::save(const unsigned short& t) {
	save((unsigned char)((t) & 0xFF));
	save((unsigned char)((t >> 8) & 0xFF));
}
template <>
void dmBinIStream::load(unsigned short& t) {
	unsigned char buf;
	load(buf); t = buf;
	load(buf); t += buf << 8;
}

#define __STREAM_SIGNED(Type,Max,Middle) template <> \
void dmBinOStream::save(const Type& t) { \
	if( t > 0 ) { \
		save(static_cast<unsigned Type>(t)); \
		return; \
	} \
	unsigned Type value = Max;\
	value -= -t;\
	value += 1;\
	save(value);\
} \
\
template<> \
void dmBinIStream::load(Type& t) { \
	unsigned Type buffer; \
	load(buffer); \
	if( buffer <= Middle ) { \
		t = buffer; \
		return; \
	} \
	t = buffer-1-Max; \
	return; \
}

__STREAM_SIGNED(long long, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF)
__STREAM_SIGNED(int, 0xFFFFFFFF, 0x7FFFFFFF)
__STREAM_SIGNED(short, 0xFFFF, 0x7FFF)


static unsigned int floatToUnsigned(float t) {
	unsigned int data;
	if (t == 0) {
		data = 0;
	}
	else if (std::isnan(t)) {
		data = 2143289344;
	}
	else if (t == std::numeric_limits<float>::infinity()) {
		data = 2139095040;
	}
	else if (t == -std::numeric_limits<float>::infinity()) {
		data = 4286578688;
	}
	else {
		bool minus = false;
		if (t < 0) { minus = true; t *= -1; }
		int exponent;
		float fraction = std::frexp(t, &exponent);

		data = ((unsigned int)(fraction * (1 << 24)) & 0x7FFFFF);
		data |= (exponent + 126) << 23;
		if (minus) { data |= 0x80000000; }
	}
	return data;
}
static void unsignedToFloat(unsigned int data, float& result) {
	if (data == 0) {
		result = 0.0;
	}
	else if (data == 2143289344) {
		result = std::numeric_limits<float>::quiet_NaN();
	}
	else if (data == 2139095040) {
		result = std::numeric_limits<float>::infinity();
	}
	else if (data == 4286578688) {
		result = -std::numeric_limits<float>::infinity();
	}
	else {
		bool minus = data >> 31;
		int exponent = ((data >> 23) & 0xFF) - 126;
		float fraction = double((data & 0x7FFFFF) | 0x800000) / (1 << 24);
		if (minus) fraction *= -1.0;
		result = ldexp(fraction, exponent);
	}
}
template <>
void dmBinOStream::save(const float& t) {
	save(floatToUnsigned(t));
}

template <>
void dmBinIStream::load(float& t) {
	unsigned int buffer;
	load(buffer);
	unsignedToFloat(buffer, t);
}


static unsigned long long doubleToUnsigned(double t) {
	unsigned long long data;
	if (t == 0) {
		data = 0;
	}
	else if (isnan(t)) {
		data = 9221120237041090560ul;
	}
	else if (t == std::numeric_limits<double>::infinity()) {
		data = 9218868437227405312ul;
	}
	else if (t == -std::numeric_limits<double>::infinity()) {
		data = 18442240474082181120ul;
	}
	else {
		bool minus = false;
		if (t < 0) { minus = true; t *= -1; };
		int exponent;
		double fraction = std::frexp(t, &exponent);

		data = ((unsigned long long)(fraction * (1ull << 53))) & 0xFFFFFFFFFFFFF;
		data |= (unsigned long long)(exponent + 1022) << 52;
		if (minus) { data |= 0x8000000000000000; }
	}

	return data;
}
static void unsignedToDouble(unsigned long long data, double& result) {
	if (data == 0) {
		result = 0.0;
	}
	else if (data == 9221120237041090560ul) {
		result = std::numeric_limits<double>::quiet_NaN();
	}
	else if (data == 9218868437227405312ul) {
		result = std::numeric_limits<double>::infinity();
	}
	else if (data == 18442240474082181120ul) {
		result = -std::numeric_limits<double>::infinity();
	}
	else {
		bool minus = data >> 63;
		int exponent = ((data >> 52) & 0x7FF) - 1022;
		double fraction = double((data & 0xFFFFFFFFFFFFF) | 0x10000000000000) / (1ull << 53);
		if (minus) fraction *= -1.0;
		result = std::ldexp(fraction, exponent);
	}
}

template <>
void dmBinOStream::save(const double& t) {
	save(doubleToUnsigned(t));
}

template <>
void dmBinIStream::load(double& t) {
	unsigned long long buffer;
	load(buffer);
	unsignedToDouble(buffer, t);
}

