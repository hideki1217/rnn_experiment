#pragma once

#include <random>

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
}  // namespace mynn