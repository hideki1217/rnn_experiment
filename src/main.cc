#include <memory>
#include <random>
#include <vector>

#include "mynn.h"

using namespace mynn;

void print_vec(const double* p, int n) {
  printf("%f", p[0]);
  for (int j = 1; j < n; j++) printf(", %f", p[j]);
  printf("\n");
}

void nn() {
  int size[] = {1, 100, 1};

  auto l0 = std::make_unique<T[]>(size[0]);
  auto layer1 =
      linear::Layer(size[0], size[1], std::make_unique<linear::Grad>(0.01));
  auto act1 = act::Layer<c1func::Tanh>(size[1]);
  auto l1 = std::make_unique<T[]>(size[1]);
  auto layer2 =
      linear::Layer(size[1], size[2], std::make_unique<linear::Grad>(0.01));
  auto l2 = std::make_unique<T[]>(size[2]);

  auto f = [](T x) { return x * x; };

  auto x = std::make_unique<T[]>(10000);
  auto y = std::make_unique<T[]>(10000);
  {
    std::mt19937 engin(42);

    std::normal_distribution<> norm(0.0, 1.0);
    for (int i = 0; i < 10000; i++) {
      x[i] = norm(engin);
      y[i] = f(x[i]);
    }

    for (int i = 0; i < layer1.x_n * layer1.y_n; i++) {
      layer1.weight[i] = norm(engin);
    }
    for (int i = 0; i < layer2.x_n * layer2.y_n; i++) {
      layer2.weight[i] = norm(engin);
    }
  }

  auto score = [](T y_act, T y_true) {
    return std::pow(y_act - y_true, 2) / 2.0;
  };
  auto score_diff = [](T y_act, T y_true) { return (y_act - y_true); };
  std::vector<int> idxs(100000);
  for (int i = 0; i < idxs.size(); i++) idxs[i] = i % 10000;
  std::random_shuffle(idxs.begin(), idxs.end());

  T prev = 1e6;
  for (int epoch = 0; epoch < idxs.size(); epoch++) {
    int idx = idxs[epoch];

    l0[0] = x[idx];
    layer1.forward(l0.get(), l1.get());
    act1.forward(l1.get(), l1.get());
    layer2.forward(l1.get(), l2.get());

    auto res = score(l2[0], y[idx]);
    printf("%d: %.9lf\n", epoch, res);
    if (std::abs(res - prev) < 1e-9) {
      // break;
    } else {
      prev = res;
    }
    l2[0] = score_diff(l2[0], y[idx]);
    layer2.learn_backward(l2.get(), l1.get());
    act1.learn_backward(l1.get(), l1.get());
    layer1.learn_backward(l1.get(), l0.get());
  }
}

void nn_RMSProp() {
  int size[] = {1, 100, 1};

  auto l0 = std::make_unique<T[]>(size[0]);
  auto layer1 = linear::Layer(size[0], size[1],
                              std::make_unique<linear::Momentum>(0.01, 0.2));
  auto act1 = act::Layer<c1func::Tanh>(size[1]);
  auto l1 = std::make_unique<T[]>(size[1]);
  auto layer2 = linear::Layer(size[1], size[2],
                              std::make_unique<linear::Momentum>(0.01, 0.2));
  auto l2 = std::make_unique<T[]>(size[2]);

  auto f = [](T x) { return x * x; };

  auto x = std::make_unique<T[]>(10000);
  auto y = std::make_unique<T[]>(10000);
  {
    std::mt19937 engin(42);

    std::normal_distribution<> norm(0.0, 1.0);
    for (int i = 0; i < 10000; i++) {
      x[i] = norm(engin);
      y[i] = f(x[i]);
    }

    for (int i = 0; i < layer1.x_n * layer1.y_n; i++) {
      layer1.weight[i] = norm(engin);
    }
    for (int i = 0; i < layer2.x_n * layer2.y_n; i++) {
      layer2.weight[i] = norm(engin);
    }
  }

  auto score = [](T y_act, T y_true) {
    return std::pow(y_act - y_true, 2) / 2.0;
  };
  auto score_diff = [](T y_act, T y_true) { return (y_act - y_true); };
  std::vector<int> idxs(100000);
  for (int i = 0; i < idxs.size(); i++) idxs[i] = i % 10000;
  std::random_shuffle(idxs.begin(), idxs.end());

  T prev = 1e6;
  for (int epoch = 0; epoch < idxs.size(); epoch++) {
    int idx = idxs[epoch];

    l0[0] = x[idx];
    layer1.forward(l0.get(), l1.get());
    act1.forward(l1.get(), l1.get());
    layer2.forward(l1.get(), l2.get());

    auto res = score(l2[0], y[idx]);
    printf("%d: %.9lf\n", epoch, res);
    if (std::abs(res - prev) < 1e-9) {
      // break;
    } else {
      prev = res;
    }
    l2[0] = score_diff(l2[0], y[idx]);
    layer2.learn_backward(l2.get(), l1.get());
    act1.learn_backward(l1.get(), l1.get());
    layer1.learn_backward(l1.get(), l0.get());
  }
}

void rnn() {
  const int N = 10;

  auto l0 = std::make_unique<double[]>(N);
  auto layer = linear::Layer(N, N, std::make_unique<linear::Grad>(0.01));
  auto act = act::Layer<c1func::Tanh>(N);
  auto l1 = std::make_unique<double[]>(N);

  {
    std::mt19937 engin(42);

    std::normal_distribution<> norm(0.0, 0.9);
    for (int i = 0; i < layer.x_n * layer.y_n; i++) {
      layer.weight[i] = norm(engin);
    }
    for (int i = 0; i < layer.y_n; i++) {
      layer.bias[i] = norm(engin);
    }

    for (int i = 0; i < N; i++) l0[i] = norm(engin);
  }

  for (int i = 0; i < 10; i++) {
    print_vec(l0.get(), N);

    layer.forward(l0.get(), l1.get());
    act.forward(l1.get(), l1.get());

    std::copy(l1.get(), l1.get() + N, l0.get());
  }
  print_vec(l0.get(), N);
}

int main() {
  // nn();
  nn_RMSProp();
}