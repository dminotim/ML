#pragma once
#include <vector>

class dmBinIStream
{
public:
	dmBinIStream(const std::vector<char>& data)
		:m_data(data), m_offset(0)
	{
	}
	virtual ~dmBinIStream() {
	}

	template <typename T>
	dmBinIStream& operator>>(T& obj) {
		load(obj);
		return *this;
	}

	template <typename T>
	void load(T& obj);

	virtual void load(char& ch)
	{
		ch = m_data[m_offset++];
	};
	inline void load(unsigned char& ch) { load(reinterpret_cast<char&>(ch)); }
private:
	size_t m_offset;
	std::vector<char> m_data;
};

class dmBinOStream
{
public:
	virtual ~dmBinOStream()
	{
	}

	template <typename T>
	dmBinOStream& operator<<(const T& obj) {
		save(obj);
		return *this;
	}

	template <typename T>
	void save(const T& obj);

	inline void save(const unsigned char& c)
	{ 
		save(static_cast<char>(c));
	}

	void save(const char& c)
	{
		m_values.push_back(c);
	}
//private:
	std::vector<char> m_values;
};