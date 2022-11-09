#pragma once

#include <cassert>
#include <memory>

#include "_mynn.h"

namespace mynn {
namespace linear {
class Layer;

class Opt {
 public:
  virtual void apply(Layer& lay, const T* dy) = 0;
};

class Layer {
 public:
  Layer(int batch_n, int x_n, int y_n, std::unique_ptr<Opt> optimizer)
      : batch_n(batch_n), x_n(x_n), y_n(y_n), opt(std::move(optimizer)) {
    weight = std::make_unique<T[]>(x_n * y_n);
    bias = std::make_unique<T[]>(y_n);
    _x = std::make_unique<T[]>(batch_n * x_n);
  }

  void forward(const T* x, T* y) {
    std::copy_n(x, batch_n * x_n, _x.get());

    for (int b = 0; b < batch_n; b++) {
      for (int i = 0; i < y_n; i++) {
        T out_i = 0;
        for (int j = 0; j < x_n; j++) {
          out_i += weight[i * x_n + j] * x[b * x_n + j];
        }
        y[b * y_n + i] = out_i + bias[i];
      }
    }
  }
  void learn(const T* dy) { opt->apply(*this, dy); }
  void backward(const T* dy, T* dx) {
    for (int j = 0; j < x_n; j++) {
      T dx_sum = 0;
      for (int b = 0; b < batch_n; b++) {
        for (int i = 0; i < y_n; i++) {
          dx_sum += dy[b * y_n + i] * weight[i * x_n + j];
        }
      }
      dx[j] = dx_sum / batch_n;
    }
  }
  void learn_backward(const T* dy, T* dx) {
    learn(dy);
    backward(dy, dx);
  }

  const int batch_n, x_n, y_n;
  std::unique_ptr<T[]> weight;
  std::unique_ptr<T[]> bias;
  std::unique_ptr<Opt> opt;
  std::unique_ptr<T[]> _x;
};

class Grad : public Opt {
 public:
  Grad(T alpha) : alpha(alpha) {}
  void apply(Layer& lay, const T* dy) {
    for (int i = 0; i < lay.y_n; i++) {
      for (int j = 0; j < lay.x_n; j++) {
        T g_sum = 0;
        for (int b = 0; b < lay.batch_n; b++) {
          g_sum += dy[b * lay.y_n + i] * lay._x[b * lay.x_n + j];
        }
        lay.weight[i * lay.x_n + j] -= alpha * g_sum / lay.batch_n;
      }
    }

    for (int i = 0; i < lay.y_n; i++) {
      lay.bias[i] -= alpha * dy[i];
    }
  }

  T alpha;
};

class Momentum : public Opt {
  /**
   * @brief v_{t+1} = beta * v_{t} + (1 - beta) * G
   * w_{t+1} = w_t - alpha * v_{t+1}
   */
 public:
  Momentum(T alpha, T beta) : alpha(alpha), beta(beta) {
    assert(0 <= beta && beta <= 1);
  }
  void apply(Layer& lay, const T* dy) {
    if (!v_W) init(lay.x_n, lay.y_n);

    for (int i = 0; i < lay.y_n; i++) {
      for (int j = 0; j < lay.x_n; j++) {
        T g_sum = 0;
        for (int b = 0; b < lay.batch_n; b++) {
          g_sum += dy[b * lay.y_n + i] * lay._x[b * lay.x_n + j];
        }
        const T g = g_sum / lay.batch_n;
        const T v = (beta * v_W[i * lay.x_n + j] + (1 - beta) * g);

        lay.weight[i * lay.x_n + j] -= alpha * v;

        v_W[i * lay.x_n + j] = v;
      }
    }

    for (int i = 0; i < lay.y_n; i++) {
      const T g = dy[i];
      const T v = beta * v_b[i] + (1 - beta) * g;

      lay.bias[i] -= alpha * v;

      v_b[i] = v;
    }
  }

  T alpha;
  T beta;
  std::unique_ptr<T[]> v_W;
  std::unique_ptr<T[]> v_b;

 private:
  void init(int x_n, int y_n) {
    v_W = std::make_unique<T[]>(x_n * y_n);
    v_b = std::make_unique<T[]>(y_n);

    std::fill_n(v_W.get(), x_n * y_n, T(0));
    std::fill_n(v_b.get(), y_n, T(0));
  }
};

class RMSProp : public Opt {
 public:
  RMSProp(T alpha, T beta) : alpha(alpha), beta(beta) {
    assert(0 <= beta && beta <= 1);
  }

  void apply(Layer& lay, const T* dy) {
    if (!v_W) init(lay.x_n, lay.y_n);

    for (int i = 0; i < lay.y_n; i++) {
      for (int j = 0; j < lay.x_n; j++) {
        T g_sum = 0;
        for (int b = 0; b < lay.batch_n; b++) {
          g_sum += dy[b * lay.y_n + i] * lay._x[b * lay.x_n + j];
        }
        const T g = g_sum / lay.batch_n;
        const T v = (beta * v_W[i * lay.x_n + j] + (1 - beta) * g * g);

        lay.weight[i * lay.x_n + j] -= alpha * g / std::sqrt(v + 1e-12);

        v_W[i * lay.x_n + j] = v;
      }
    }

    for (int i = 0; i < lay.y_n; i++) {
      const T g = dy[i];
      const T v = beta * v_b[i] + (1 - beta) * g * g;

      lay.bias[i] -= alpha * g / std::sqrt(v + 1e-12);

      v_b[i] = v;
    }
  }

  T alpha;
  T beta;
  std::unique_ptr<T[]> v_W;
  std::unique_ptr<T[]> v_b;

 private:
  void init(int x_n, int y_n) {
    v_W = std::make_unique<T[]>(x_n * y_n);
    v_b = std::make_unique<T[]>(y_n);

    std::fill_n(v_W.get(), x_n * y_n, T(0));
    std::fill_n(v_b.get(), y_n, T(0));
  }
};
}  // namespace linear
}  // namespace mynn
