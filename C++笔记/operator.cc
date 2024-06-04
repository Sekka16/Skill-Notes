#include <iostream>
#include <print>

template <typename T> struct Point {
  T x;
  T y;

  Point(T xCoord, T yCoord) : x(xCoord), y(yCoord) {}

  Point operator+(const Point &other) const {
    return Point(x + other.x, y + other.y);
  }

  Point operator*(const Point &other) const {
    return Point(x * other.x, y * other.y);
  }

  Point operator-(const Point &other) const {
    return Point(x - other.x, y - other.y);
  }

  friend std::ostream &operator<<(std::ostream &os, const Point &p) {
    os << "(" << p.x << ", " << p.y << ")";
    return os;
  }
};

template <typename T, typename T2>
Point<T> operator*(const T2 &scalar_p, const Point<T> &p) {
  T scalar = static_cast<T>(scalar_p);
  return Point<T>(p.x * scalar, p.y * scalar);
}

template <typename T> Point<T> operator+(const T &scalar, const Point<T> &p) {
  return Point<T>(scalar + p.x, scalar + p.y);
}

template <typename T> Point<T> operator-(const T &scalar, const Point<T> &p) {
  return Point<T>(scalar - p.x, scalar - p.y);
}

template <typename T> Point<T> operator*(const Point<T> &p, const T &scalar) {
  return Point<T>(p.x * scalar, p.y * scalar);
}

template <typename T> Point<T> operator+(const Point<T> &p, const T &scalar) {
  return Point<T>(scalar + p.x, scalar + p.y);
}

template <typename T> Point<T> operator-(const Point<T> &p, const T &scalar) {
  return Point<T>(scalar - p.x, scalar - p.y);
}

int main() {

  Point<int> a(1, 2);
  Point<int> b(1, 2);
  Point<short> c(1, 2);
  Point<unsigned int> d(1, 2);
  Point<long> e(1, 2);
  Point<float> f1(1.0f, 2.0f);
  Point<float> f2(2.5f, 3.0f);
  Point<float> f3(-2.0f, 0.0f);

  Point<float> F = 2.0f * f1 + f2 - 4 * f3;

  std::cout << F << std::endl;

  return 0;
}
