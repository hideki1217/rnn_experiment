#pragma once

#include <cblas.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <vector>

namespace mynn {

using T = double;

namespace resource {
struct Param {
  Param(int size, bool with_grad = true) : size(size) {
    _v = std::make_unique<T[]>(size);
    if (with_grad) _grad = std::make_unique<T[]>(size);
  }

  void free_grad() { std::move(_grad); }
  T* v() { return _v.get(); }
  T* grad() { return _grad.get(); }

  int size;
  std::unique_ptr<T[]> _v;
  std::unique_ptr<T[]> _grad;
};
}  // namespace resource

namespace dataproc {}

namespace model {
class Evaluable {
 public:
  virtual void reset() {}
  virtual void forward(T* y) = 0;
  virtual T* input() = 0;
};
class LayerBase : public Evaluable {
 public:
  virtual void backward(const T* dy) = 0;
};

class ModelBase {
 public:
  virtual int y_size() = 0;
  virtual int x_size() = 0;
  virtual std::unique_ptr<LayerBase> create(int batch) = 0;
  virtual std::unique_ptr<Evaluable> create_for_eval(int batch) {
    return down_cast<Evaluable>(create(batch));
  };
};

class Affine : public ModelBase {
 public:
  Affine(resource::Param& w, resource::Param& b, bool carry = false)
      : x_n(w.size / b.size), y_n(b.size), carry(carry), w(w), b(b) {
    assert(w.size % b.size == 0);
  }
  Affine(ModelBase& parent, resource::Param& w, resource::Param& b)
      : Affine(w, b) {
    assert(parent.y_size() == x_n);
  }

  struct Layer : public LayerBase {
    Layer(int b_n, Affine& model) : b_n(b_n), model(model) {
      _x = std::make_unique<T[]>(b_n * model.x_n);
    }

    void forward(T* y) {
      for (int b = 0; b < b_n; b++) {
        for (int i = 0; i < model.y_n; i++) {
          y[b * model.y_n + i] = model.b._v[i];
        }
      }
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, b_n, model.y_n,
                  model.x_n, 1.0, _x.get(), model.x_n, model.w.v(), model.x_n,
                  1.0, y, model.y_n);
    }
    void backward(const T* dy) {
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, model.y_n, model.x_n,
                  b_n, 1.0, dy, model.y_n, _x.get(), model.x_n,
                  model.carry ? 1.0 : 0.0, model.w.grad(), model.x_n);

      if (!model.carry) std::fill_n(model.b.grad(), model.y_n, T(0));
      for (int b = 0; b < b_n; b++) {
        for (int i = 0; i < model.y_n; i++) {
          model.b._grad[i] += dy[b * model.y_n + i];
        }
      }

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, b_n, model.x_n,
                  model.y_n, 1.0, dy, model.y_n, model.w.v(), model.x_n, 0.0,
                  _x.get(), model.x_n);
    }
    T* input() { return _x.get(); }

    int b_n;
    Affine& model;
    std::unique_ptr<T[]> _x;
  };
  std::unique_ptr<LayerBase> create(int b_n) {
    return std::unique_ptr<LayerBase>(new Layer(b_n, *this));
  }
  int y_size() { return y_n; }
  int x_size() { return x_n; }

  bool carry;  // w, b への勾配を足し込むか？
  int x_n, y_n;
  resource::Param &w, &b;
};

template <typename F>
class Act : public ModelBase {
 public:
  Act(int n, F f = F()) : n(n), f(f) {}
  Act(ModelBase& parent, F f = F()) : Act(parent.y_size(), f) {}

  struct Layer : public LayerBase {
    Layer(int b_n, Act& model) : b_n(b_n), model(model) {
      _x = std::make_unique<T[]>(b_n * model.n);
    }
    void forward(T* y) {
      for (int i = 0; i < b_n * model.n; i++) {
        y[i] = model.f(_x[i]);
      }
    }
    void backward(const T* dy) {
      for (int i = 0; i < b_n * model.n; i++) {
        _x[i] = dy[i] * model.f.d(_x[i]);
      }
    }

    T* input() { return _x.get(); }

    int b_n;
    Act& model;
    std::unique_ptr<T[]> _x;
  };
  std::unique_ptr<LayerBase> create(int b_n) {
    return std::unique_ptr<LayerBase>(new Layer(b_n, *this));
  }
  int y_size() { return n; }
  int x_size() { return n; }

  int n;
  F f;
};

class SoftMax : public ModelBase {
 public:
  SoftMax(int n) : n(n) {}
  SoftMax(ModelBase& before) : SoftMax(before.y_size()){};

  struct Layer : LayerBase {
    Layer(int b_n, SoftMax& model) : b_n(b_n), model(model) {
      _x = std::make_unique<T[]>(b_n * model.n);
      _y = std::make_unique<T[]>(b_n * model.n);
    }

    void forward(T* y) {
      for (int b = 0; b < b_n; b++) {
        T max = max_n(&_x[b * model.n], model.n);
        for (int i = 0; i < model.n; i++) {
          y[b * model.n + i] = std::exp(_x[b * model.n + i] - max);
        }

        T _sum = 1.0 / (sum_n(&y[b * model.n], model.n) + 1e-12);
        for (int i = 0; i < model.n; i++) {
          y[b * model.n + i] *= _sum;
        }
      }

      std::copy_n(y, b_n * model.n, _y.get());
    }
    void backward(const T* dy) {
      for (int b = 0; b < b_n; b++) {
        T sum = 0;
        for (int j = 0; j < model.n; j++) {
          sum += dy[b * model.n + j] * _y[b * model.n + j];
        }

        for (int i = 0; i < model.n; i++) {
          _x[b * model.n + i] =
              (dy[b * model.n + i] - sum) * _y[b * model.n + i];
        }
      }
    }

    T* input() { return _x.get(); }

    int b_n;
    std::unique_ptr<T[]> _x;
    std::unique_ptr<T[]> _y;
    SoftMax& model;

   private:
    T max_n(const T* x, int n) {
      T res = 0;
      for (int i = 0; i < n; i++) {
        res = std::max(res, x[i]);
      }
      return res;
    }
    T sum_n(const T* x, int n) {
      T res = 0;
      for (int i = 0; i < n; i++) {
        res += x[i];
      }
      return res;
    }
  };

  std::unique_ptr<LayerBase> create(int b_n) {
    return std::unique_ptr<LayerBase>(new Layer(b_n, *this));
  }
  int y_size() { return n; }
  int x_size() { return n; }

  int n;
};

}  // namespace model
namespace c1func {
class Tanh {
 public:
  T operator()(T x) const { return std::tanh(x); }
  T d(T x) const { return 1.0 / std::pow(std::cosh(x), 2); }
};

class Relu {
 public:
  T operator()(T x) const { return (x > 0) ? x : 0; }
  T d(T x) const { return (x > 0) ? 1 : 0; }
};

class Identity {
 public:
  T operator()(T x) const { return x; }
  T d(T x) const { return 1; }
};
}  // namespace c1func
namespace policy {
class Grad {
 public:
  Grad(T alpha) : alpha(alpha) {}
  Grad(const Grad& rhs) : Grad(rhs.alpha) {}
  void add_target(resource::Param& param) { targets.emplace_back(param); }
  template <typename... Rest>
  void add_target(resource::Param& first, Rest&... rest) {
    add_target(first);
    add_target(rest...);
  }

  void update() {
    for (auto& target : targets) {
      int size = target.get().size;
      T* x = target.get().v();
      const T* dx = target.get().grad();

      for (int i = 0; i < size; i++) {
        x[i] -= alpha * dx[i];
      }
    }
  }

  T alpha;
  std::vector<std::reference_wrapper<resource::Param>> targets;
};
class RMSProp {
 public:
  RMSProp(T alpha, T beta) : alpha(alpha), beta(beta) {
    assert(0 <= beta && beta <= 1);
  }
  RMSProp(const RMSProp& rhs) : RMSProp(rhs.alpha, rhs.beta) {}
  void add_target(resource::Param& param) {
    auto _v = std::make_unique<T[]>(param.size);
    std::fill_n(_v.get(), param.size, T(0));

    targets.emplace_back(param);
    _vs.emplace_back(std::move(_v));
  }
  template <typename... Rest>
  void add_target(resource::Param& first, Rest&... rest) {
    add_target(first);
    add_target(rest...);
  }
  void update() {
    for (int i = 0; i < targets.size(); i++) {
      auto size = targets[i].get().size;
      auto x = targets[i].get().v();
      auto dx = targets[i].get().grad();
      auto _v = _vs[i].get();

      for (int j = 0; j < size; j++) {
        const T v = beta * _v[j] + (1 - beta) * dx[j] * dx[j];

        x[j] -= alpha * dx[j] / std::sqrt(v + 1e-12);

        _v[j] = v;
      }
    }
  }

  T alpha;
  T beta;
  std::vector<std::reference_wrapper<resource::Param>> targets;
  std::vector<std::unique_ptr<T[]>> _vs;
};
class Momentum {
 public:
  Momentum(T alpha, T beta) : alpha(alpha), beta(beta) {
    assert(0 <= beta && beta <= 1);
  }
  Momentum(const Momentum& rhs) : Momentum(rhs.alpha, rhs.beta) {}
  void add_target(resource::Param& param) {
    auto _v = std::make_unique<T[]>(param.size);
    std::fill_n(_v.get(), param.size, T(0));

    targets.emplace_back(param);
    _vs.emplace_back(std::move(_v));
  }
  template <typename... Rest>
  void add_target(resource::Param& first, Rest&... rest) {
    add_target(first);
    add_target(rest...);
  }

  void update() {
    for (int i = 0; i < targets.size(); i++) {
      int size = targets[i].get().size;
      T* x = targets[i].get().v();
      const T* dx = targets[i].get().grad();
      T* _v = _vs[i].get();

      for (int i = 0; i < size; i++) {
        T v = beta * _v[i] - (1 - beta) * alpha * dx[i];
        x[i] += v;
        _v[i] = v;
      }
    }
  }

  T alpha, beta;
  std::vector<std::reference_wrapper<resource::Param>> targets;
  std::vector<std::unique_ptr<T[]>> _vs;
};
class Adam {
 public:
  Adam(T alpha, T beta0, T beta1) : alpha(alpha), beta0(beta0), beta1(beta1) {
    assert(0 <= beta0 && beta0 <= 1);
    assert(0 <= beta1 && beta1 <= 1);
  }
  Adam(const Adam& rhs) : Adam(rhs.alpha, rhs.beta0, rhs.beta1) {}
  void add_target(resource::Param& param) {
    auto _v0 = std::make_unique<T[]>(param.size);
    std::fill_n(_v0.get(), param.size, T(0));
    auto _v1 = std::make_unique<T[]>(param.size);
    std::fill_n(_v1.get(), param.size, T(0));

    targets.emplace_back(param);
    _v0s.emplace_back(std::move(_v0));
    _v1s.emplace_back(std::move(_v1));
  }
  template <typename... Rest>
  void add_target(resource::Param& first, Rest&... rest) {
    add_target(first);
    add_target(rest...);
  }

  void update() {
    for (int i = 0; i < targets.size(); i++) {
      int size = targets[i].get().size;
      T* x = targets[i].get().v();
      const T* dx = targets[i].get().grad();
      T* _v0 = _v0s[i].get();
      T* _v1 = _v1s[i].get();

      for (int i = 0; i < size; i++) {
        const T v0 = beta0 * _v0[i] + (1 - beta0) * dx[i];
        const T v1 = beta1 * _v1[i] + (1 - beta1) * dx[i] * dx[i];

        x[i] -= alpha * v0 / std::sqrt(v1 + 1e-12);

        _v0[i] = v0;
        _v1[i] = v1;
      }
    }
  }

  T alpha, beta0, beta1;
  std::vector<std::reference_wrapper<resource::Param>> targets;
  std::vector<std::unique_ptr<T[]>> _v0s, _v1s;
};
}  // namespace policy
namespace loss_func {
class CrossEntropy {
 public:
  CrossEntropy(int n) : n(n) {}
  CrossEntropy(model::ModelBase& parent) : CrossEntropy(parent.y_size()) {}

  struct Impl {
    Impl(int b_n, CrossEntropy& model) : b_n(b_n), model(model) {
      _x = std::make_unique<T[]>(b_n * model.n);
      _correct = std::make_unique<int[]>(b_n);
    }

    void reset() {}
    void forward(T* y) {
      T res = 0;
      for (int b = 0; b < b_n; b++) {
        res -= std::log(_x[b * model.n + _correct[b]]);
      }
      *y = res / b_n;
    }
    void backward() {
      for (int b = 0; b < b_n; b++) {
        for (int i = 0; i < model.n; i++) {
          if (i == _correct[b]) {
            _x[b * model.n + _correct[b]] =
                -1.0 / (_x[b * model.n + _correct[b]] + 1e-12) / b_n;
          } else {
            _x[b * model.n + i] = 0.0;
          }
        }
      }
    }
    T* input() { return _x.get(); }
    int* correct() { return _correct.get(); }

    int b_n;
    std::unique_ptr<int[]> _correct;
    std::unique_ptr<T[]> _x;
    CrossEntropy& model;
  };

  std::unique_ptr<Impl> create(int b_n) {
    return std::make_unique<Impl>(b_n, *this);
  }
  int y_size() { return n; }
  int x_size() { return n; }

  int n;
};
class MSE {
 public:
  MSE(int n) : n(n) {}
  MSE(model::ModelBase& parent) : MSE(parent.y_size()) {}

  struct Impl {
    Impl(int b_n, MSE& model) : b_n(b_n), model(model) {
      _x = std::make_unique<T[]>(b_n * model.n);
      _correct = std::make_unique<T[]>(b_n * model.n);
    }

    void reset() {}
    void forward(T* y) {
      T res = 0, _n = 1.0 / model.n;
      for (int i = 0; i < model.n * b_n; i++) {
        res += std::pow(y[i] - _correct[i], 2) * _n;
      }
      *y = res / 2.0;
    }
    void backward() {
      T _n = 1.0 / (model.n * b_n);
      for (int i = 0; i < model.n * b_n; i++) {
        _x[i] = (_x[i] - _correct[i]) * _n;
      }
    }
    T* input() { return _x.get(); }
    T* correct() { return _correct.get(); }

    int b_n;
    std::unique_ptr<T[]> _correct;
    std::unique_ptr<T[]> _x;
    MSE& model;
  };

  std::unique_ptr<Impl> create(int b_n) {
    return std::make_unique<Impl>(b_n, *this);
  }
  int y_size() { return n; }
  int x_size() { return n; }

  int n;
};
}  // namespace loss_func
namespace lr {
template <typename Opt>
class ReduceOnPleatou {
 public:
  ReduceOnPleatou(Opt& optimizer, int patience, T lr_ratio)
      : optimizer(optimizer), patience(patience), lr_ratio(lr_ratio) {}

  bool step(T score) {
    if (score >= min_score) {
      count++;
      if (count >= patience) {
        min_score = score;
        count = 0;

        optimizer.alpha *= lr_ratio;
        return true;
      }
    } else {
      min_score = score;
      count = 0;
    }
    return false;
  }
  T current_lr() { return optimizer.alpha; }

  int patience;
  T lr_ratio;

 private:
  Opt& optimizer;
  T min_score = 1e10;
  int count = 0;
};
}  // namespace lr
}  // namespace mynn