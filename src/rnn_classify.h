#pragma once

#include <exception>
#include <memory>
#include <random>

#include "common.h"
#include "mynn.h"

using namespace mynn;

template <typename V>
void mvmul(int m, int n, const V *w, int ldw, const V *v, int incv, V *r,
           int incr) {
  V::NotImplemetedError();
}

template <>
void mvmul<double>(int m, int n, const double *w, int ldw, const double *v,
                   int incv, double *r, int incr) {
  cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, w, ldw, v, incv, 0.0, r,
              incr);
}

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

    for (int i = 0; i < inner_dim; i++) inner[i] = noise(engine);
    for (int i = 0; i < inner_dim; i++) inner[i] += means[c * inner_dim + i];

    cblas_dgemv(CblasRowMajor, CblasNoTrans, dim, inner_dim, 1.0, w.get(),
                inner_dim, inner, 1, 0.0, res, 1);

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

class MultiClass {
 public:
  virtual int random_gen(T *res) = 0;
  virtual int gen(T *res, int c) = 0;
  virtual int num_class() = 0;
  virtual int res_size() = 0;
};

class BaseData : public MultiClass {
 public:
  BaseData(int dim, int class_n, int cluster_n, T noise_scale, int seed)
      : dim(dim),
        class_n(class_n),
        cluster_n(cluster_n),
        engine(std::mt19937(seed)),
        noise_scale(noise_scale),
        noise(std::normal_distribution<>(0.0, noise_scale)),
        prior_dist(std::uniform_int_distribution<>(0, cluster_n - 1)) {
    means = std::make_unique<T[]>(cluster_n * dim);
    labels = std::make_unique<int[]>(cluster_n);

    init_means();
  }

  int num_class() { return class_n; }
  int res_size() { return dim; }

  int random_gen(T *res) {
    int c = prior_dist(engine);  // std::rand() % cluster_n;
    return gen(res, c);
  }

  int gen(T *res, int c) {
    for (int i = 0; i < dim; i++) res[i] = noise(engine);
    for (int i = 0; i < dim; i++) res[i] += means[c * dim + i];
    return labels[c];
  }

  int dim, class_n, cluster_n;
  T noise_scale;
  std::mt19937 engine;
  std::normal_distribution<T> noise;
  std::uniform_int_distribution<int> prior_dist;
  std::unique_ptr<T[]> means;
  std::unique_ptr<int[]> labels;

 private:
  void init_means() {
    T minimam =
        noise_scale * 10;  // 標準偏差の5倍の距離は保つ　外に出る確率が1e-7

    T tmp[dim];
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
          for (int j = 0; j < dim; j++) {
            distance += std::pow(means[i * dim + j] - tmp[j], 2.0);
          }
          distance = std::sqrt(distance);

          if (distance <= minimam) {
            flag = true;
            break;
          }
        }
      }

      std::copy_n(tmp, dim, &means[c * dim]);
      labels[c] = c % class_n;
    }
  }
};

class LinEmbed : public BaseData {
 public:
  LinEmbed(int dim, std::unique_ptr<T[]> &&w, int inner_dim, int class_n,
           int cluster_n, T noise_scale, int seed)
      : BaseData(inner_dim, class_n, cluster_n, noise_scale, seed),
        embed_dim(dim),
        w(std::move(w)) {}
  int res_size() { return embed_dim; }

  int gen(T *res, int c) {
    T inner[dim];
    BaseData::gen(inner, c);

    mvmul(embed_dim, dim, w.get(), dim, inner, 1, res, 1);
    return labels[c];
  }

  std::unique_ptr<T[]> w;
  int embed_dim;
};

class RandomEmbed : public LinEmbed {
 public:
  RandomEmbed(int dim, int inner_dim, int class_n, int cluster_n, T noise_scale,
              int seed)
      : LinEmbed(dim, make_w(dim, inner_dim), inner_dim, class_n, cluster_n,
                 noise_scale, seed) {}

 private:
  std::unique_ptr<T[]> make_w(int m, int n) {
    w = std::make_unique<T[]>(m * n);
    auto tmp = std::make_unique<T[]>(n * m);
    random_orthogonal(n, m, tmp.get(), engine);
    transpose(n, m, tmp.get(), w.get());
    return std::move(w);
  }
};