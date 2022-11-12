#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <random>

namespace mynn {

using T = double;

enum class Layers {
  Linear,
  Act,
};
class LayerBase {
 public:
  virtual Layers kind() const = 0;
  virtual int y_size() const = 0;
  virtual int x_size() const = 0;
  virtual int batch_size() const = 0;
  virtual void forward(const T* x, T* y) = 0;
  virtual void backward(const T* dy, T* dx) = 0;
};

namespace linear {

class Layer : public LayerBase {
 public:
  Layer(int b_n, int x_n, int y_n) : b_n(b_n), x_n(x_n), y_n(y_n) {
    weight = std::make_unique<T[]>(x_n * y_n);
    bias = std::make_unique<T[]>(y_n);
    d_weight = std::make_unique<T[]>(x_n * y_n);
    d_bias = std::make_unique<T[]>(y_n);

    _x = std::make_unique<T[]>(b_n * x_n);
  }
  Layer(LayerBase& before, int y_n)
      : Layer(before.batch_size(), before.y_size(), y_n){};
  Layers kind() const { return Layers::Linear; }
  int x_size() const { return x_n; }
  int y_size() const { return y_n; }
  int batch_size() const { return b_n; }
  void forward(const T* x, T* y);
  void backward(const T* dy, T* dx);
  int weight_size() { return x_n * y_n; }
  int bias_size() { return y_n; }

  int b_n, x_n, y_n;
  std::unique_ptr<T[]> weight;
  std::unique_ptr<T[]> bias;
  std::unique_ptr<T[]> d_weight;
  std::unique_ptr<T[]> d_bias;

 private:
  std::unique_ptr<T[]> _x;

 public:
  void xivier(int seed);
  void he(int seed);
};
}  // namespace linear

namespace act {

template <typename F>
class Layer : public LayerBase {
 public:
  Layer(int b_n, int n) : b_n(b_n), n(n), act(F()) {
    _x = std::make_unique<T[]>(b_n * n);
  }
  Layer(LayerBase& before) : Layer(before.batch_size(), before.y_size()){};

  Layers kind() const { return Layers::Act; }
  virtual int y_size() const { return n; }
  virtual int x_size() const { return y_size(); }
  virtual int batch_size() const { return b_n; }
  virtual void forward(const T* x, T* y) {
    std::copy_n(x, b_n * n, _x.get());
    for (int i = 0; i < b_n * n; i++) {
      y[i] = act(x[i]);
    }
  }
  virtual void backward(const T* dy, T* dx) {
    for (int i = 0; i < b_n * n; i++) {
      dx[i] = dy[i] * act.d(_x[i]);
    }
  }

  int b_n, n;
  F act;

 private:
  std::unique_ptr<T[]> _x;
};
}  // namespace act
namespace c1func {
class Tanh {
 public:
  T operator()(T x) const { return std::tanh(x); }
  T d(T x) const { return 1.0 / std::pow(std::cosh(x), 2); }
};

class Relu {
 public:
  T operator()(T x) const { return (x > 0) ? x : 0; }
  T d(T x) const { return (x > 0) ? 1 : 0; }
};

class Identity {
 public:
  T operator()(T x) const { return x; }
  T d(T x) const { return 1; }
};
}  // namespace c1func
namespace opt {
class Grad {
 public:
  Grad(T alpha) : alpha(alpha) {}
  void regist(T* x, const T* dx, int size) {
    this->x = x;
    this->dx = dx;
    this->size = size;
  }

  void update() {
    for (int i = 0; i < size; i++) {
      x[i] -= alpha * dx[i];
    }
  }

  T alpha;
  T* x;
  const T* dx;
  int size;
};

class Momentum {
  /**
   * @brief v_{t+1} = beta * v_{t} + (1 - beta) * G
   * w_{t+1} = w_t - alpha * v_{t+1}
   */
 public:
  Momentum(T alpha, T beta) : alpha(alpha), beta(beta) {
    assert(0 <= beta && beta <= 1);
  }
  void regist(T* x, const T* dx, int size) {
    assert(!_v);
    init(size);
    this->x = x;
    this->dx = dx;
    this->size = size;
  }
  void update() {
    for (int i = 0; i < size; i++) {
      T v = (beta * _v[i] + (1 - beta) * dx[i]);
      x[i] -= alpha * v;
      _v[i] = v;
    }
  }

  T alpha;
  T beta;
  T* x;
  const T* dx;
  int size;
  std::unique_ptr<T[]> _v;

 private:
  void init(int size) {
    _v = std::make_unique<T[]>(size);

    std::fill_n(_v.get(), size, T(0));
  }
};

class RMSProp {
 public:
  RMSProp(T alpha, T beta) : alpha(alpha), beta(beta) {
    assert(0 <= beta && beta <= 1);
  }
  void regist(T* x, const T* dx, int size) {
    assert(!_v);
    init(size);
    this->x = x;
    this->dx = dx;
    this->size = size;
  }
  void update() {
    for (int i = 0; i < size; i++) {
      const T v = beta * _v[i] + (1 - beta) * dx[i] * dx[i];

      x[i] -= alpha * dx[i] / std::sqrt(v + 1e-12);

      _v[i] = v;
    }
  }

  T alpha;
  T beta;
  T* x;
  const T* dx;
  int size;
  std::unique_ptr<T[]> _v;

 private:
  void init(int size) {
    _v = std::make_unique<T[]>(size);

    std::fill_n(_v.get(), size, T(0));
  }
};

class Adam {
 public:
  Adam(T alpha, T beta0, T beta1) : alpha(alpha), beta0(beta0), beta1(beta1) {
    assert(0 <= beta0 && beta0 <= 1);
    assert(0 <= beta1 && beta1 <= 1);
  }
  void regist(T* x, const T* dx, int size) {
    assert(!_v);
    init(size);
    this->x = x;
    this->dx = dx;
    this->size = size;
  }
  void update() {
    for (int i = 0; i < size; i++) {
      const T v0 = beta0 * _v0[i] + (1 - beta0) * dx[i];
      const T v1 = beta1 * _v1[i] + (1 - beta1) * dx[i] * dx[i];

      x[i] -= alpha * v0 / std::sqrt(v1 + 1e-12);

      _v0[i] = v0;
      _v1[i] = v1;
    }
  }

  T alpha;
  T beta0;
  T beta1;
  T* x;
  const T* dx;
  int size;
  std::unique_ptr<T[]> _v0;  // momentum
  std::unique_ptr<T[]> _v1;  // Rmsprop
 private:
  void init(int size) {
    _v0 = std::make_unique<T[]>(size);
    _v1 = std::make_unique<T[]>(size);

    std::fill_n(_v0.get(), size, T(0));
    std::fill_n(_v1.get(), size, T(0));
  }
};
}  // namespace opt
}  // namespace mynn