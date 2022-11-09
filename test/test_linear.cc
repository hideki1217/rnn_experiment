#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <vector>

#include "mynn.h"

void test_forward() {
  std::vector<double> in(1000), out(1000);

  auto layer = LinearLayer<double>(1000, 1000);
  for (int i = 0; i < 1000; i++) {
    for (int j = 0; j < 1000; j++) {
      layer.weight[i * 1000 + j] = (i == j) * 3;
    }
  }
  std::fill_n(layer.bias.get(), layer.y_n, 1);

  for (int i = 0; i < in.size(); i++) in[i] = i;
  layer.forward(in.data(), out.data());
  for (int i = 0; i < out.size(); i++)
    assert(std::abs(out[i] - (in[i] * 3 + 1)) < 1e-6);
}

void test_learn() {
  double in, out;

  auto f = [](double x) { return 3.5 * x + 4; };
  auto score = [](double x, double x_true) { return std::pow(x - x_true, 2); };
  auto score_diff = [](double x, double x_true) { return 2 * (x - x_true); };

  auto layer = LinearLayer<double>(1, 1);

  std::mt19937 engine(42);
  std::normal_distribution<> norm(0.0, 4.0);
  layer.weight[0] = norm(engine);
  layer.bias[0] = norm(engine);
  layer.set_param(0.01);

  for (int i = 0; i < 1000; i++) {
    in = norm(engine);
    layer.forward(&in, &out);

    printf("%d: act = %f, true = %f, score = %f\n", i, out, f(in),
           score(out, f(in)));
    out = score_diff(out, f(in));
    layer.learn(&out);
  }

  assert(std::abs(layer.weight[0] - 3.5) < 1e-4);
  assert(std::abs(layer.bias[0] - 4) < 1e-4);
}

int main() {
  test_forward();
  test_learn();
}