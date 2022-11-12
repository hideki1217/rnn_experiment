#include "mynn.h"

#include <cblas.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>

void mynn::linear::Layer::forward(const T* x, T* y) {
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
              _x.get(), x_n, weight.get(), x_n, 1.0, y, y_n);
}

void mynn::linear::Layer::backward(const T* dy, T* dx) {
  std::fill_n(d_weight.get(), x_n * y_n, T(0));
  // for (int b = 0; b < b_n; b++) {
  //   for (int i = 0; i < y_n; i++) {
  //     for (int j = 0; j < x_n; j++) {
  //       d_weight[i * x_n + j] += dy[b * y_n + i] * _x[b * x_n + j];
  //     }
  //   }
  // }
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, y_n, x_n, b_n, 1.0, dy,
              y_n, _x.get(), x_n, 1.0, d_weight.get(), x_n);

  std::fill_n(d_bias.get(), y_n, T(0));
  for (int b = 0; b < b_n; b++) {
    for (int i = 0; i < y_n; i++) {
      d_bias[i] += dy[b * y_n + i];
    }
  }

  std::fill_n(dx, x_n * b_n, T(0));
  // for (int b = 0; b < b_n; b++) {
  //   for (int i = 0; i < y_n; i++) {
  //     for (int j = 0; j < x_n; j++) {
  //       dx[b * x_n + j] += dy[b * y_n + i] * weight[i * x_n + j];
  //     }
  //   }
  // }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, b_n, x_n, y_n, 1.0, dy,
              y_n, weight.get(), x_n, 1.0, dx, x_n);
}

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
