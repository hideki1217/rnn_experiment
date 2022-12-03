#pragma once

#include <memory>
#include <random>

#include "common.h"
#include "mynn.h"

using namespace mynn;

class DataProc {
 public:
  DataProc(int dim, int inner_dim, int class_n, int cluster_n, T noise_scale,
           int seed)
      : dim(dim),
        inner_dim(inner_dim),
        class_n(class_n),
        cluster_n(cluster_n),
        engine(std::mt19937(seed)),
        noise_scale(noise_scale),
        noise(std::normal_distribution<>(0.0, noise_scale)),
        rand(std::uniform_int_distribution<>(0, cluster_n - 1)) {
    means = std::make_unique<T[]>(cluster_n * inner_dim);
    labels = std::make_unique<int[]>(cluster_n);
    w = std::make_unique<T[]>(dim * inner_dim);

    init_w();
    init_means();
  }

  int random_gen(T *res) {
    int c = rand(engine);  // std::rand() % cluster_n;
    return gen(res, c);
  }

  int gen(T *res, int c) {
    T inner[inner_dim];
    for (int i = 0; i < inner_dim; i++) {
      inner[i] = means[c * inner_dim + i] + noise(engine);
    }

    for (int i = 0; i < dim; i++) {
      T reg = 0;
      for (int j = 0; j < inner_dim; j++) {
        reg += w[i * inner_dim + j] * inner[j];
      }
      res[i] = reg;
    }

    return labels[c];
  }

  int dim, inner_dim, class_n, cluster_n;
  T noise_scale;
  std::mt19937 engine;
  std::normal_distribution<T> noise;
  std::uniform_int_distribution<int> rand;
  std::unique_ptr<T[]> means;
  std::unique_ptr<int[]> labels;
  std::unique_ptr<T[]> w;

 private:
  void init_w() {
    auto tmp = std::make_unique<T[]>(inner_dim * dim);
    random_orthogonal(inner_dim, dim, tmp.get(), engine);
    transpose(inner_dim, dim, tmp.get(), w.get());
  }
  void init_means() {
    T minimam =
        noise_scale * 10;  // 標準偏差の5倍の距離は保つ　外に出る確率が1e-7

    T tmp[inner_dim];
    const T L = 4.0;
    std::uniform_real_distribution<> unif(-L / 2, L / 2);
    for (int c = 0; c < cluster_n; c++) {
      bool flag = true;
      while (flag) {
        // uniform sample from inner_dim's hypercube
        for (auto &x : tmp) x = unif(engine);

        flag = false;

        for (int i = 0; i < c; i++) {
          T distance = 0;
          for (int j = 0; j < inner_dim; j++) {
            distance += std::pow(means[i * inner_dim + j] - tmp[j], 2.0);
          }
          distance = std::sqrt(distance);

          if (distance <= minimam) {
            flag = true;
            break;
          }
        }
      }

      std::copy_n(tmp, inner_dim, &means[c * inner_dim]);
      labels[c] = c % class_n;
    }
  }
};