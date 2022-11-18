#include "mynn.h"

#include <cblas.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>

void mynn::linear::Layer::xivier(int seed) {
  std::mt19937 engine(seed);
  std::normal_distribution<> dist(0.0, 1.0 / std::sqrt(x_n));

  for (int i = 0; i < y_n * x_n; i++) {
    weight[i] = dist(engine);
  }

  std::fill_n(bias.get(), y_n, T(0));
}

void mynn::linear::Layer::he(int seed) {
  std::mt19937 engine(seed);
  std::normal_distribution<> dist(0.0, std::sqrt(2.0 / x_n));

  for (int i = 0; i < y_n * x_n; i++) {
    weight[i] = dist(engine);
  }

  std::fill_n(bias.get(), y_n, T(0));
}

void mynn::impl::Linear::forward(const T* x, T* y) {
  /**
   * @brief x is (b_n * x_n, )
   * y is (b_n * y_n)
   */
  std::copy_n(x, b_n * x_n, _x.get());

  for (int b = 0; b < b_n; b++) {
    for (int i = 0; i < y_n; i++) {
      y[b * y_n + i] = bias[i];
    }
  }
  // for (int b = 0; b < b_n; b++) {
  //   for (int i = 0; i < y_n; i++) {
  //     for (int j = 0; j < x_n; j++) {
  //       y[b * y_n + i] += weight[i * x_n + j] * _x[b * x_n + j];
  //     }
  //   }
  // }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, b_n, y_n, x_n, 1.0,
              _x.get(), x_n, weight, x_n, 1.0, y, y_n);
}
void mynn::impl::Linear::backward(const T* dy, T* dx) {
  /**
   * @brief 微分を足し込む
   */

  // for (int b = 0; b < b_n; b++) {
  //   for (int i = 0; i < y_n; i++) {
  //     for (int j = 0; j < x_n; j++) {
  //       d_weight[i * x_n + j] += dy[b * y_n + i] * _x[b * x_n + j];
  //     }
  //   }
  // }
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, y_n, x_n, b_n, 1.0, dy,
              y_n, _x.get(), x_n, 1.0, d_weight, x_n);

  for (int b = 0; b < b_n; b++) {
    for (int i = 0; i < y_n; i++) {
      d_bias[i] += dy[b * y_n + i];
    }
  }

  // for (int b = 0; b < b_n; b++) {
  //   for (int i = 0; i < y_n; i++) {
  //     for (int j = 0; j < x_n; j++) {
  //       dx[b * x_n + j] += dy[b * y_n + i] * weight[i * x_n + j];
  //     }
  //   }
  // }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, b_n, x_n, y_n, 1.0, dy,
              y_n, weight, x_n, 1.0, dx, x_n);
}

mynn::custom::RNN::RNN(int b_n, int n, int depth)
    : b_n(b_n), n(n), depth(depth) {
  weight = std::make_unique<T[]>(n * n);
  bias = std::make_unique<T[]>(n);
  d_weight = std::make_unique<T[]>(n * n);
  d_bias = std::make_unique<T[]>(n);

  state = std::make_unique<T[]>(b_n * n);
  _cache = std::make_unique<T[]>(b_n * n);

  for (int t = 0; t < depth; t++) {
    _lins.push_back(impl::Linear(b_n, n, n, weight.get(), bias.get(),
                                 d_weight.get(), d_bias.get()));
    _acts.push_back(act::Layer<c1func::Tanh>(b_n, n));
  }
}
void mynn::custom::RNN::forward(const T* x, T* y) {
  /**
   * @brief x is [b_n][n], y is [b_n][n]
   */
  for (int t = 0; t < depth; t++) {
    _lins[t].forward(state.get(), state.get());
    if (t == 0) {
      for (int i = 0; i < b_n * n; i++) {
        state[i] += x[i];
      }
    }

    _acts[t].forward(state.get(), state.get());
  }

  std::copy_n(state.get(), b_n * n, y);
}
void mynn::custom::RNN::backward(const T* dy, T* dx) {
  std::copy_n(dy, b_n * n, dx);

  std::fill_n(d_weight.get(), weight_size(), T(0));
  std::fill_n(d_bias.get(), bias_size(), T(0));
  for (int t = depth - 1; t >= 0; t--) {
    _acts[t].backward(dx, _cache.get());

    std::fill_n(dx, b_n * n, T(0));
    _lins[t].backward(_cache.get(), dx);
  }
}