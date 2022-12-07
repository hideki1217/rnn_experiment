#pragma once

#include <cassert>
#include <memory>
#include <random>

template <typename V>
int argmax_n(const V* a, int n) {
  V max = -1000000;
  int res = -1;
  for (int i = 0; i < n; i++) {
    if (max < a[i]) {
      max = a[i];
      res = i;
    }
  }
  return res;
}

template <typename V>
void inner_product(int m, int n, const V* w, const V* v, V* out) {
  for (int i = 0; i < m; i++) {
    V reg = 0;
    for (int j = 0; j < n; j++) {
      reg += w[i * n + j] * v[j];
    }
    out[i] = reg;
  }
}

template <typename V>
V vvmul(int n, const V* w, const V* v) {
  V reg = 0;
  for (int j = 0; j < n; j++) {
    reg += w[j] * v[j];
  }
  return reg;
}

template <typename V>
V l2norm(int n, const V* v) {
  V reg = 0;
  for (int i = 0; i < n; i++) reg += v[i] * v[i];
  return reg;
}

template <typename V, typename F, typename... A>
void map(F f, V* out, int n, const A*... w) {
  for (int i = 0; i < n; i++) {
    out[i] = f(w[i]...);
  }
}

template <typename V, typename RandEngine>
void random_orthogonal(int m, int n, V* w, RandEngine&& engine) {
  std::normal_distribution<> dist(0.0, 1.0);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      w[i * n + j] = dist(engine);
    }

    for (int k = 0; k < i; k++) {
      auto inner = vvmul(n, &w[k * n], &w[i * n]);
      map([=](V x, V y) { return y - x * inner; }, &w[i * n], n, &w[k * n],
          &w[i * n]);
    }
    auto _len = 1.0 / std::sqrt(l2norm(n, &w[i * n]));
    map([=](V x) { return x * _len; }, &w[i * n], n, &w[i * n]);
  }
}

template <typename V>
void transpose(int m, int n, const V* src, V* dst) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[j * m + i] = src[i * n + j];
    }
  }
}

template <typename B, typename D>
std::unique_ptr<B> down_cast(std::unique_ptr<D>&& ptr) {
  return std::unique_ptr<B>(static_cast<B*>(ptr.release()));
}

template <typename D, typename B>
std::unique_ptr<D> dynamic_ptr_cast(std::unique_ptr<B>&& ptr) {
  auto p = ptr.release();
  auto res = dynamic_cast<D*>(p);
  if (res) {
    return std::unique_ptr<D>(res);
  } else {
    abort();
  }
}

struct Acc {
  Acc() { reset(); }
  void reset() { correct = n = 0; }
  void ok() {
    correct++;
    n++;
  }
  void err() { n++; }
  void step(int res) {
    if (res)
      ok();
    else
      err();
  }
  double result() { return (double)correct / n; }

  int correct;
  int n;
};

class Logger {
 public:
  Logger(std::string path) : path(path) {
    file = std::fopen(path.c_str(), "w");
    if (file == nullptr) {
      printf("Logger: cannot file open\n");
      abort();
    }
  }
  ~Logger() { std::fclose(file); }
  template <typename... Args>
  void print(const char* format, Args const&... args) {
    std::fprintf(file, format, args...);
  }
  void print(const char* format) { std::fprintf(file, "%s", format); }

  std::string path;
  FILE* file;
};
