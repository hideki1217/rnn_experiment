#include <memory>

template <typename T>
class LinearLayer {
 public:
  LinearLayer(int x_n, int y_n) : x_n(x_n), y_n(y_n) {
    weight = std::make_unique<T[]>(x_n * y_n);
    bias = std::make_unique<T[]>(y_n);
    _x = std::make_unique<T[]>(x_n);
  }

  void set_param(T alpha) { this->alpha = alpha; }
  void forward(const T* x, T* y) {
    std::copy_n(x, x_n, _x.get());
    for (int i = 0; i < y_n; i++) {
      T out_i = 0;
      for (int j = 0; j < x_n; j++) {
        out_i += weight[i * x_n + j] * x[j];
      }
      y[i] = out_i + bias[i];
    }
  }
  void learn(const T* dy) {
    for (int i = 0; i < y_n; i++) {
      for (int j = 0; j < x_n; j++) {
        weight[i * x_n + j] -= alpha * (dy[i] * _x[j]);
      }
    }

    for (int i = 0; i < y_n; i++) {
      bias[i] -= alpha * dy[i];
    }
  }
  void backward(const T* dy, T* dx) {
    for (int j = 0; j < x_n; j++) {
      T dx_j = 0;
      for (int i = 0; i < y_n; i++) {
        dx_j += dy[i] * weight[i * x_n + j];
      }
      dx[j] = dx_j;
    }
    // std::fill_n(dx, x_n, T(0));
    // for (int i = 0; i < y_n; i++) {
    //   T in_i = dy[i];
    //   for (int j = 0; j < x_n; j++) {
    //     dx[j] += in_i * weight[i * x_n + j];
    //   }
    // }
  }
  void learn_backward(const T* dy, T* dx) {
    learn(dy);
    backward(dy, dx);
  }

  const int x_n, y_n;
  std::unique_ptr<T[]> weight;
  std::unique_ptr<T[]> bias;
  T alpha = 1e-2;

 private:
  std::unique_ptr<T[]> _x;
};