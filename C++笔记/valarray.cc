#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <new>
#include <type_traits>

constexpr int VECTOR_LENGTH = 10;

template <typename T> class vector {
  static_assert(std::is_same<T, int>::value || std::is_same<T, float>::value,
                "Invalid type for vector, must be int or float");

private:
  typedef T value_type;

public:
  value_type *_data;
  size_t _size;
  size_t _capacity;

public:
  // 构造函数
  vector() : _data(NULL), _size(0), _capacity(VECTOR_LENGTH) {
    _data = new (std::nothrow) value_type[_capacity];

    if (!_data) {
      throw std::bad_alloc();
    }
  }

  explicit vector(size_t _size)
      : _data(NULL), _size(_size), _capacity(VECTOR_LENGTH) {
    while (_size > _capacity) {
      _capacity *= 2;
    }

    _data = new (std::nothrow) value_type[_capacity];
    if (!_data) {
      throw std::bad_alloc();
    }
  }

  vector(std::initializer_list<value_type> values)
      : _size(values.size()), _capacity(values.size()) {
    _data = new (std::nothrow) value_type[_capacity];

    if (!_data) {
      throw std::bad_alloc();
    }

    std::copy(values.begin(), values.end(), _data);
  }

  // 拷贝构造函数(深拷贝)
  vector(const vector &vec) {
    _size = vec._size;
    _capacity = vec._capacity;

    _data = new (std::nothrow) value_type[_capacity];
    if (!_data) {
      throw std::bad_alloc();
    }

    std::copy(vec._data, vec._data + vec._size, _data);
  }

  // 析构函数
  ~vector() {
    delete[] _data;
    _data = NULL;
    _size = 0;
    _capacity = 0;
  }

  // =
  vector &operator=(const vector &vec) {
    if (this == &vec)
      return *this;

    vector tmp(vec);

    delete[] _data;
    _data = tmp._data;
    _size = tmp._size;
    _capacity = tmp._capacity;

    tmp._data = NULL;
    tmp._size = 0;
    tmp._capacity = 0;

    return *this;
  }

  // +
  vector operator+(const vector &vec) {
    size_t common_size = std::min(_size, vec._size);
    vector result(common_size);

    std::transform(_data, _data + common_size, vec._data, result._data,
                   std::plus<value_type>());

    return result;
  }

  // // -(minus)
  // vector operator-(const vector &vec) {
  //   size_t common_size = std::min(_size, vec._size);
  //   vector result(common_size);
  //
  //   std::transform(_data, _data + common_size, vec._data, result._data,
  //                  std::minus<value_type>());
  //   return result;
  // }

  // -(neg)
  vector operator-() const {
    vector result(_size);
    std::transform(_data, _data + _size, result._data,
                   std::negate<value_type>());
    return result;
  }

  vector abs() {
    vector result(_size);
    std::transform(_data, _data + _size, result._data,
                   [](value_type val) { return std::abs(val); });
    return result;
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const vector<value_type> &vec) {
    os << "[";
    for (size_t i = 0; i < vec._size; ++i) {
      os << vec._data[i];
      if (i < vec._size - 1) {
        os << ", ";
      }
    }
    os << "]";
    return os;
  }
};

// +
template <typename T1, typename T2>
auto operator+(const vector<T1> &vec1,
               const vector<T2> &vec2) -> vector<decltype(T1() + T2())> {
  using VecType = decltype(T1() + T2());
  size_t common_size = std::min(vec1._size, vec2._size);
  vector<VecType> result(common_size);

  std::transform(vec1._data, vec1._data + common_size, vec2._data, result._data,
                 std::plus<VecType>());

  return result;
}
// -
template <typename T1, typename T2>
auto operator-(const vector<T1> &vec1,
               const vector<T2> &vec2) -> vector<decltype(T1() + T2())> {
  using VecType = decltype(T1() + T2());
  size_t common_size = std::min(vec1._size, vec2._size);
  vector<VecType> result(common_size);

  std::transform(vec1._data, vec1._data + common_size, vec2._data, result._data,
                 std::minus<VecType>());

  return result;
}

template <typename T1, typename T2>
vector<T1> operator+(const T2 &scalar, const vector<T1> &vec) {
  using VecType = typename std::conditional<
      std::is_same<decltype(T1() + T2()), double>::value, float,
      decltype(T1() + T2())>::type;
  vector<VecType> result(vec._size);

  std::transform(vec._data, vec._data + vec._size, result._data,
                 [=](VecType val) { return val + scalar; });

  return result;
}

template <typename T1, typename T2>
vector<T1> operator+(const vector<T1> &vec, const T2 &scalar) {
  using VecType = typename std::conditional<
      std::is_same<decltype(T1() + T2()), double>::value, float,
      decltype(T1() + T2())>::type;
  vector<VecType> result(vec._size);

  std::transform(vec._data, vec._data + vec._size, result._data,
                 [=](VecType val) { return val + scalar; });

  return result;
}

template <typename T1, typename T2>
vector<T1> operator-(const T2 &scalar, const vector<T1> &vec) {
  using VecType = typename std::conditional<
      std::is_same<decltype(T1() + T2()), double>::value, float,
      decltype(T1() + T2())>::type;
  vector<VecType> result(vec._size);

  std::transform(vec._data, vec._data + vec._size, result._data,
                 [=](VecType val) { return scalar - val; });

  return result;
}

template <typename T1, typename T2>
vector<T1> operator-(const vector<T1> &vec, const T2 &scalar) {
  using VecType = typename std::conditional<
      std::is_same<decltype(T1() + T2()), double>::value, float,
      decltype(T1() + T2())>::type;
  vector<VecType> result(vec._size);

  std::transform(vec._data, vec._data + vec._size, result._data,
                 [=](VecType val) { return val - scalar; });

  return result;
}

// vector abs(vector) {}

int main() {
  vector<int> x{1, 2, 3, 4, 5};
  vector<float> y{3.0f, 5.2f, -1.0f, 0.0f, 2.0f};
  // vector<double> y;

  vector<int> x1, x2, x3, x4, x5;
  x1 = x + x;
  x2 = x - x;
  x3 = -x;
  x4 = x3.abs();
  x5 = x1 + 2;

  vector<float> z;
  // ..
  z = -(x - y + 2) - 4.5;

  std::cout << x1 << std::endl;
  std::cout << x2 << std::endl;
  std::cout << x3 << std::endl;
  std::cout << x4 << std::endl;
  std::cout << x5 << std::endl;
  std::cout << x + y << std::endl;

  std::cout << z << std::endl;

  return 0;
}
