#pragma once

#include <cblas.h>
#include <lapack.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace ryap {
using T = double;

static std::vector<T> identity(int n) {
  std::vector<T> res(n * n);
  std::fill(res.begin(), res.end(), T(0));
  for (int i = 0; i < n; i++) res[i * n + i] = T(1);
  return res;
}

class F {
 public:
  void operator()(const T* x, T* y);
  void d(const T* x, T* dF);
  int x_size();
};

template <typename System>
std::vector<T> spectoram(System&& target, int m = 1, T eps = 1e-4,
                         int seed = 42) {
  int n = target.x_size();
  std::mt19937 engine(seed);
  std::normal_distribution<> dist(0.0, 1.0);

  std::vector<T> prev(n);
  std::vector<T> eigval(n, 1e9);
  auto is_converged = [&]() {
    std::sort(eigval.begin(), eigval.end());
    for (int i = 0; i < eigval.size(); i++) {
      if (std::abs(eigval[i] - prev[i]) >= eps) return false;
    }
    return true;
  };

  std::vector<T> res(n * m);
  for (int i = 0; i < m; i++) {
    int t = 1;
    std::vector<T> x_t(n);
    for (auto& x : x_t) x = 10.0 * dist(engine);
    auto W = identity(n);
    // 一定時間ごとに、Wを定数倍小さくする処理のときの定数倍のlogを足し込んでおく
    T normalize_logsum = T(0);
    std::vector<T> W_t_W(n * n);
    std::vector<T> df_t(n * n);
    std::vector<T> work(n * 4);
    const int lwork = work.size();
    int info = 0;
    do {
      std::copy(eigval.begin(), eigval.end(), prev.begin());

      target.d(x_t.data(), df_t.data());
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                  df_t.data(), n, W.data(), n, 0.0, W_t_W.data(), n);
      std::copy(W_t_W.begin(), W_t_W.end(), W.begin());

      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, n, 1.0,
                  W.data(), n, W.data(), n, 0.0, W_t_W.data(), n);

      dsyev_("N", "U", &n, W_t_W.data(), &n, eigval.data(), work.data(), &lwork,
             &info, n, n);
      std::sort(eigval.begin(), eigval.end());
      std::transform(eigval.begin(), eigval.end(), eigval.begin(),
                     [](auto x) { return std::abs(x); });  // これはいいのか？
      if (info) throw std::runtime_error("error: dsyev_()");

      {  // 標準化
        T _beta = 1.0 / std::accumulate(eigval.begin(), eigval.end(), T(0));
        normalize_logsum += std::log(_beta);
        for (int i = 0; i < n * n; i++) {
          W[i] *= _beta;
        }
        for (int i = 0; i < n; i++) eigval[i] *= _beta;
      }

      for (int i = 0; i < eigval.size(); i++) {
        eigval[i] = (normalize_logsum + std::log(eigval[i])) / (2 * t);
      }

      t++;
      target(x_t.data(), x_t.data());
    } while (!is_converged());

    std::copy_n(eigval.data(), n, &res[i * n]);
  }

  return res;
}

template <typename System>
std::vector<T> spectoram_n(System&& target, int rank_n, int m = 1, T eps = 1e-4,
                           int seed = 42) {
  int n = target.x_size();
  std::mt19937 engine(seed);
  std::normal_distribution<> dist(0.0, 1.0);

  std::vector<T> prev(rank_n);
  std::vector<T> eigval(n, 1e9);
  auto is_converged = [&]() {
    std::sort(&eigval[0], &eigval[rank_n]);
    for (int i = 0; i < rank_n; i++) {
      if (std::abs(eigval[i] - prev[i]) >= eps) return false;
    }
    return true;
  };

  std::vector<T> res(rank_n * m);
  for (int i = 0; i < m; i++) {
    int t = 1;
    std::vector<T> x_t(n);
    for (auto& x : x_t) x = 10.0 * dist(engine);
    auto W = identity(n);
    // 一定時間ごとに、Wを定数倍小さくする処理のときの定数倍のlogを足し込んでおく
    T normalize_logsum = T(0);
    std::vector<T> W_t_W(n * n);
    std::vector<T> df_t(n * n);

    int il = n - rank_n + 1, iu = n;
    double vl, vu;
    int ifail;
    double z;
    int ldz = 1;
    std::vector<T> work(n * 8);
    std::vector<int> iwork(n * 5);
    int lwork = work.size();
    int info = 0;
    do {
      std::copy_n(&eigval[0], rank_n, prev.begin());

      target.d(x_t.data(), df_t.data());
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                  df_t.data(), n, W.data(), n, 0.0, W_t_W.data(), n);
      std::copy(W_t_W.begin(), W_t_W.end(), W.begin());

      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, n, 1.0,
                  W.data(), n, W.data(), n, 0.0, W_t_W.data(), n);

      lwork = -1;
      LAPACK_dsyevx("N", "I", "U", &n, W_t_W.data(), &n, &vl, &vu, &il, &iu,
                    &eps, &rank_n, eigval.data(), &z, &ldz, work.data(), &lwork,
                    iwork.data(), &ifail, &info);
      lwork = (int)work[0];
      work.resize(lwork);
      LAPACK_dsyevx("N", "I", "U", &n, W_t_W.data(), &n, &vl, &vu, &il, &iu,
                    &eps, &rank_n, eigval.data(), &z, &ldz, work.data(), &lwork,
                    iwork.data(), &ifail, &info);

      std::sort(&eigval[0], &eigval[rank_n]);
      std::transform(&eigval[0], &eigval[rank_n], &eigval[0],
                     [](auto x) { return std::abs(x); });  // これはいいのか？
      if (info) throw std::runtime_error("error: dsyev_()");

      {  // 標準化
        T _beta = 1.0 / std::accumulate(&eigval[0], &eigval[rank_n], T(0));
        normalize_logsum += std::log(_beta);
        for (int i = 0; i < n * n; i++) {
          W[i] *= _beta;
        }
        for (int i = 0; i < rank_n; i++) eigval[i] *= _beta;
      }

      for (int i = 0; i < rank_n; i++) {
        eigval[i] = (normalize_logsum + std::log(eigval[i])) / (2 * t);
      }

      t++;
      target(x_t.data(), x_t.data());
    } while (!is_converged());

    std::copy_n(eigval.data(), rank_n, &res[i * rank_n]);
  }

  return res;
}

class RNN : F {
 public:
  RNN(int n) : n(n) {
    weight.resize(n * n);
    bias.resize(n);
    _y.resize(n);
  }
  void operator()(const T* x, T* y) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, weight.data(), n, x, 1,
                0.0, _y.data(), 1);
    for (int i = 0; i < n; i++) {
      y[i] = std::tanh(_y[i] + bias[i]);
    }
  }
  void d(const T* x, T* dF) {
    operator()(x, _y.data());
    std::copy(weight.begin(), weight.end(), dF);
    for (int i = 0; i < n; i++) {
      T d_tanh_i = 1.0 - _y[i] * _y[i];
      for (int j = 0; j < n; j++) {
        dF[i * n + j] *= d_tanh_i;
      }
    }
  }
  int x_size() { return n; }

  int n;
  std::vector<T> weight;
  std::vector<T> bias;

 private:
  std::vector<T> _y;
};
}  // namespace ryap