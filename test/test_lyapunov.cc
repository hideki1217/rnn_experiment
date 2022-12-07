#include <cblas.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "lyapunov.h"

using namespace ryap;

#define dprint_float(exp) printf(#exp " = %lf\n", exp);

#define assert(expr)                                 \
  if (!(expr)) {                                     \
    printf("%s:%d " #expr "\n", __FILE__, __LINE__); \
    abort();                                         \
  }

class Linear : F {
 public:
  Linear(int n)
      : n(n), w(std::make_unique<T[]>(n * n)), _x(std::make_unique<T[]>(n)) {}

  void operator()(const T* x, T* y) {
    std::copy_n(x, n, _x.get());
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, w.get(), n, _x.get(), 1,
                0.0, y, 1);
  }
  void d(const T* x, T* dF) { std::copy_n(w.get(), n * n, dF); }
  int x_size() { return n; }

  int n;
  std::unique_ptr<T[]> w;
  std::unique_ptr<T[]> _x;
};

int main() {
  const int n = 10;
  const T eps = 0.00001;
  const T g = 1, ro = 0.01;
  const int seed = 41;

  std::mt19937 engine(seed);

  Linear target(n);
  {
    T scale = g / std::sqrt(n);
    std::normal_distribution<> dist(0.0, scale);

    for (int i = 0; i < n * n; i++) {
      target.w[i] = dist(engine);
    }
  }

  std::vector<T> eigvals(n);
  {
    T q[n * n];
    T r[n * n];
    T _w[n * n];

    ryap::qr(n, target.w.get(), q, r);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, q, n,
                r, n, 0.0, _w, n);

    for (int i = 0; i < n * n; i++) {
      T lhs = target.w[i];
      T rhs = _w[i];
      assert(std::abs(lhs - rhs) < 1e-4);
    }

    std::copy_n(target.w.get(), n * n, _w);
    for (int i = 0; i < 5000; i++) {
      ryap::qr(n, _w, q, r);
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, r, n,
                  q, n, 0.0, _w, n);
    }

    for (int i = 0; i < n; i++) eigvals[i] = _w[i * n + i];
  }

  auto res = ryap::spectoram(std::move(target), 5000);
  std::sort(res.begin(), res.end());

  for (int i = 0; i < std::min(10, n); i++) {
    std::cout << res[i] << " ";
  }
  std::cout << std::endl;

  std::vector<T> ln_eigvals(n);
  for (int i = 0; i < n; i++) ln_eigvals[i] = std::log(std::abs(eigvals[i]));
  std::sort(ln_eigvals.begin(), ln_eigvals.end());
  for (int i = std::max(0, n - 10); i < n; i++) {
    std::cout << ln_eigvals[i] << " ";
  }
  std::cout << std::endl;

  // for (int i = 0; i < n; i++) assert(std::abs(res[i] - ln_eigvals[i]) <
  // 1e-3);
  std::cout << "CAUTION: Only check max LE" << std::endl;
  assert(std::abs(res[n - 1] - ln_eigvals[n - 1]) < 1e-3);
}
