#include "lyapunov.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

using namespace ryap;

#define dprint_float(exp) printf(#exp " = %lf\n", exp);

int main() {
  const int n = 200;
  const T eps = 0.00001;
  const T g = 20, ro = 0.01;
  const int seed = 41;

  std::mt19937 engine(seed);
  std::normal_distribution<> dist(0.0, 1.0);

  RNN target(n);
  {
    T scale = g / std::sqrt(n);
    target.weight = identity(n);
    for (int i = 0; i < n * n; i++) {
      target.weight[i] =
          (1 - ro) * target.weight[i] + ro * scale * dist(engine);
    }
  }

  auto res = ryap::spectoram(std::move(target), 1000);

  for (int i = 0; i < std::min(10, n); i++) {
    std::cout << res[i] << " ";
  }
  std::cout << std::endl;
}
