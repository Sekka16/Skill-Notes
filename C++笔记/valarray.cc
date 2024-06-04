#include <iostream>
#include <valarray>

constexpr int VECTOR_LENGTH = 10;

template <typename T> class vector {
public:
  typedef T value_type;
  typedef T *iterator;

private:
  value_type *_data;
  size_t _size;
  size_t _capacity;

public:
  // 构造函数
  vector() : _data(NULL), _size(0), _capacity(VECTOR_LENGTH) {}

  vector(size_t _size) : _data(NULL), _size(_size), _capacity(VECTOR_LENGTH) {}

  // 拷贝构造函数(深拷贝)
  vector(const vector &vec) {
    _size = vec._size;
    _capacity = vec._capacity;
    _data = new value_type[_capacity];

    std::copy(vec._data, vec._data + vec._size, _data);
  }

  // 析构函数
  ~vector() {
    delete[] _data;
    _data = NULL;
    _size = 0;
    _capacity = 0;
  }

  // 运算符重载
  vector &operator=(const vector &vec) {
    if (this == &vec)
      return *this;

    value_type *tmp = new value_type[vec._capacity];
    std::copy(vec._data, vec._data + vec._size, tmp);

    delete[] _data;

    _data = tmp;
    _size = vec._size;
    _capacity = vec._capacity;

    return *this;
  }

  void operator+(const vector &vec) {
    size_t common_size = std::min(_size, vec._size);

    std::transform(_data, _data + common_size, vec._data, _data,
                   std::plus<value_type>());
  }

  void operator-(const vector &vec) {
    size_t common_size = std::min(_size, vec._size);

    std::transform(_data, _data + common_size, vec._data, _data,
                   std::minus<value_type>());
  }

  void operator-() {
    std::transform(_data, _data + _size, _data, std::negate<value_type>());
  }

  // operator+, -, -(neg), =
  // operator=
};

// vector abs(vector) {}

int main() {
  vector<int> x{1, 2, 3, 4, 5};
  vector<int> y;

  vector<int> z;
  z = -(x - y + 2) - 4.5;
}
