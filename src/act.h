#include <algorithm>
#include <cmath>
#include <memory>

template <typename Act>
class ActLayer {
 public:
  using T = typename Act::Value;
  ActLayer(int n) : act(Act()), n(n) { _x = std::make_unique<T[]>(n); }

  void set_param() {}
  void forward(const T* x, T* y) {
    std::copy_n(x, n, _x.get());
    for (int i = 0; i < n; i++) {
      y[i] = act(x[i]);
    }
  }
  void learn(const T* dy) {}
  void backward(const T* dy, T* dx) {
    for (int i = 0; i < n; i++) {
      dx[i] = act.d(_x[i]) * dy[i];
    }
  }
  void learn_backward(const T* dy, T* dx) {
    learn(dy);
    backward(dy, dx);
  }

  const int n;

 private:
  const Act act;
  std::unique_ptr<T[]> _x;
};

template <typename T>
class Tanh {
 public:
  using Value = T;
  T operator()(T x) const { return std::tanh(x); }
  T d(T x) const { return 1.0 / std::pow(std::cosh(x), 2); }
};

template <typename T>
class Relu {
 public:
  using Value = T;
  T operator()(T x) const { return (x >= 0) ? x : 0; }
  T d(T x) const { return (x >= 0) ? 1 : 0; }
};

template <typename T>
class Identity {
 public:
  using Value = T;
  T operator()(T x) const { return x; }
  T d(T x) const { return 1; }
};