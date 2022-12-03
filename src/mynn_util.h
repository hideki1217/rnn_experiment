#pragma once

#include <random>
#include <tuple>

#include "mynn.h"

namespace mynn {

void affine_normal(resource::Param& w, resource::Param& b, T ro, int seed) {
  std::mt19937 engine(seed);
  std::normal_distribution<> dist(0.0, std::sqrt(ro / (w.size / b.size)));

  for (int i = 0; i < w.size; i++) {
    w._v[i] = dist(engine);
  }

  std::fill_n(b.v(), b.size, T(0));
}

void affine_normal(model::Affine& affine, T ro, int seed) {
  affine_normal(affine.w, affine.b, ro, seed);
}

namespace data {

template <typename XType, typename YType>
class DataLoader {
 public:
  DataLoader(const std::vector<XType>&& X, int x_size,
             const std::vector<YType>&& Y, int y_size, int num_batch)
      : num_batch(num_batch),
        _X(std::move(X)),
        _Y(std::move(Y)),
        x_size(x_size),
        y_size(y_size) {
    assert(_X.size() == _Y.size());

    _size = _X.size() / num_batch;
  }
  int size() { return _size; }
  XType* X() { return _X.data(); }
  YType* Y() { return _Y.data(); }

  bool load(int idx, XType* X, YType* Y) {
    if (0 <= idx && idx < _size) {
      for (int i = 0; i < num_batch; i++) {
        std::copy_n(&_X[(idx * num_batch + i) * x_size], x_size, X[i * x_size]);
        std::copy_n(&_Y[(idx * num_batch + i) * y_size], y_size, X[i * y_size]);
      }
      return true;
    }
    return false;
  }

 private:
  std::vector<XType> _X;
  int x_size;
  std::vector<YType> _Y;
  int y_size;
  int num_batch;
  int _size;
};

template <typename XType, typename YType>
DataLoader<XType, YType> make_dataloader(const std::vector<XType>&& X,
                                         int x_size,
                                         const std::vector<YType>&& Y,
                                         int y_size, int num_batch) {
  return DataLoader<XType, YType>(std::move(X), x_size, std::move(Y), y_size,
                                  num_batch);
}
}  // namespace data
}  // namespace mynn