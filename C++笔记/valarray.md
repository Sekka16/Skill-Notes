# c++中vector的实现

参考：https://blog.csdn.net/weixin_50941083/article/details/122354948

## vector的成员变量

```cpp
private:
    typedef t value_type;  // 这个并不是成员变量，主要时为了增强可读性

public:
    value_type* _data;
    size_t _size;
    size_t _capacity;
```

> 关于_capacity参数：若没有该参数，每增添一个新的元素，就需要重新申请空间存放新的元素，这会消耗较多时间，效率不高。
故而在实现的时候我们采用倍增的方式提前申请较大空间，倍增的意思就是当元素个数大于空间大小时，申请的空间将会是原空间的二倍。

## 构造函数

```cpp
  // 构造函数
  vector() : _data(null), _size(0), _capacity(vector_length) {
    _data = new (std::nothrow) value_type[_capacity];

    if (!_data) {
      throw std::bad_alloc();
    }
  }

  explicit vector(size_t _size)
      : _data(null), _size(_size), _capacity(vector_length) {
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
```

> 浅拷贝是拷贝了对象的引用，当原对象发生变化的时候，拷贝对象也跟着变化；
深拷贝是另外申请了一块内存，内容和原对象一样，更改原对象，拷贝对象不会发生变化。 

## 运算符重载

### operator=

```cpp
  vector &operator=(const vector &vec) {
    if (this == &vec)
      return *this;

    vector tmp(vec);

    delete[] _data;
    _data = tmp._data;
    _size = tmp._size;
    _capacity = tmp._capacity;

    tmp._data = null;
    tmp._size = 0;
    tmp._capacity = 0;

    return *this;
  }
```

> 在 c++ 中，this 指针是一个特殊的指针，它指向当前对象的实例。
友元函数没有 this 指针，因为友元不是类的成员，只有成员函数才有 this 指针。

### operator+

本例中实现的`vector`限定数据类型为`float`,`int`，
在处理不同类型的`vector`相加时定义为：转化成float
在处理`vector`和标量相加时定义为：推断结果类型，如果是double转化成float，否则保持原类型。

```cpp
// 向量+向量
template <typename t1, typename t2>
auto operator+(const vector<t1> &vec1,
               const vector<t2> &vec2) -> vector<decltype(t1() + t2())> {
  using vectype = decltype(t1() + t2());
  size_t common_size = std::min(vec1._size, vec2._size);
  vector<vectype> result(common_size);

  std::transform(vec1._data, vec1._data + common_size, vec2._data, result._data,
                 std::plus<vectype>());

  return result;
}

// 标量+向量
template <typename t1, typename t2>
vector<t1> operator+(const t2 &scalar, const vector<t1> &vec) {
  using vectype = typename std::conditional<
      std::is_same<decltype(t1() + t2()), double>::value, float,
      decltype(t1() + t2())>::type;
  vector<vectype> result(vec._size);

  std::transform(vec._data, vec._data + vec._size, result._data,
                 [=](vectype val) { return val + scalar; });

  return result;
}

// 向量+标量
template <typename t1, typename t2>
vector<t1> operator+(const vector<t1> &vec, const t2 &scalar) {
  using vectype = typename std::conditional<
      std::is_same<decltype(t1() + t2()), double>::value, float,
      decltype(t1() + t2())>::type;
  vector<vectype> result(vec._size);

  std::transform(vec._data, vec._data + vec._size, result._data,
                 [=](vectype val) { return val + scalar; });

  return result;
}
```

### 问题

```cpp
  // 假如类中定义
  vector operator+(const vector &vec) {
    size_t common_size = std::min(_size, vec._size);
    vector result(common_size);

    std::transform(_data, _data + common_size, vec._data, result._data,
                   std::plus<value_type>());
    return result;
  }

  // 类外定义
  template <typename t1, typename t2>
  vector<t1> operator+(const vector<t1> &vec, const t2 &scalar) {
    using vectype = typename std::conditional<
        std::is_same<decltype(t1() + t2()), double>::value, float,
        decltype(t1() + t2())>::type;
    vector<vectype> result(vec._size);

    std::transform(vec._data, vec._data + vec._size, result._data,
                   [=](vectype val) { return val + scalar; });

    return result;
  }

  // 并且你的类中有这样一个构造函数
  vector(size_t _size)
      : _data(NULL), _size(_size), _capacity(VECTOR_LENGTH) {
    while (_size > _capacity) {
      _capacity *= 2;
    }

    _data = new (std::nothrow) value_type[_capacity];
    if (!_data) {
      throw std::bad_alloc();
    }
  }
```

这样的话在执行一些语句时可能会出现警告

例如：
```cpp
x5 = x1 + 2;
```

```bash
valarray.cc:236:13: warning: ISO C++ says that these are ambiguous, even though 
the worst conversion for the first is better than the worst conversion for the s
econd:
  236 |   x5 = x1 + 2;
      |             ^
valarray.cc:186:12: note: candidate 1: ‘vector<T1> operator+(const vector<T1>&, 
const T2&) [with T1 = int; T2 = int]’
  186 | vector<T1> operator+(const vector<T1> &vec, const T2 &scalar) {
      |            ^~~~~~~~
valarray.cc:96:10: note: candidate 2: ‘vector<T> vector<T>::operator+(const vect
or<T>&) [with T = int]’
   96 |   vector operator+(const vector &vec) {
      |          ^~~~~~~~
```

造成这种警告的原因是：2被解释成构造函数的参数，编译器可能会尝试通过2这个参数创建一个`vector`，而不是将其解释为一个标量。因此会产生二义性。

**解决方法**：增加`explicit`关键字，使参数不能被进行隐式转化。

```cpp
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
```

> 在C++中，explicit关键字用于修饰单参数构造函数，防止编译器进行隐式类型转换。它的作用是禁止编译器通过隐式转换调用该构造函数。
