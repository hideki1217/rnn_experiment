
#include <fstream>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include "common.h"
#include "mynn.h"
#include "mynn_util.h"

using namespace mynn;

class Logger {
 public:
  Logger(std::string path) : path(path) {
    file = std::fopen(path.c_str(), "w");
  }
  ~Logger() { std::fclose(file); }
  template <typename... Args>
  void print(const char *format, Args const &...args) {
    std::fprintf(file, format, args...);
  }

  std::string path;
  FILE *file;
};

std::vector<int> random_index(int len, int max_index) {
  std::vector<int> res(len);
  for (int i = 0; i < len; i++) {
    res[i] = std::rand() % max_index;
  }
  return res;
}

class DataMaker {
 public:
  DataMaker(int dim, int inner_dim, int class_n, int cluster_n, T noise_scale)
      : dim(dim),
        inner_dim(inner_dim),
        class_n(class_n),
        cluster_n(cluster_n),
        engine(std::mt19937(47)),
        noise_scale(noise_scale),
        noise(std::normal_distribution<>(0.0, noise_scale)) {
    means = std::make_unique<T[]>(cluster_n * inner_dim);
    labels = std::make_unique<int[]>(cluster_n);
    w = std::make_unique<T[]>(dim * inner_dim);

    init_w();
    init_means();
  }

  int random_gen(T *res) {
    int c = std::rand() % cluster_n;
    return gen(res, c);
  }

  int gen(T *res, int c) {
    T inner[inner_dim];
    for (int i = 0; i < inner_dim; i++) {
      inner[i] = means[c * inner_dim + i] + noise(engine);
    }

    for (int i = 0; i < dim; i++) {
      T reg = 0;
      for (int j = 0; j < inner_dim; j++) {
        reg += w[i * inner_dim + j] * inner[j];
      }
      res[i] = reg;
    }

    return labels[c];
  }

  int dim, inner_dim, class_n, cluster_n;
  T noise_scale;
  std::mt19937 engine;
  std::normal_distribution<T> noise;
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
        noise_scale * 5;  // 標準偏差の5倍の距離は保つ　外に出る確率が1e-7

    T tmp[inner_dim];
    for (int c = 0; c < cluster_n; c++) {
      bool flag = true;
      while (flag) {
        for (auto &x : tmp) x = noise(engine);
        T sum = 0;
        for (auto &x : tmp) sum += x * x;
        T _l = 4.0 / std::sqrt(sum);
        for (auto &x : tmp) x *= _l;

        flag = false;

        for (int i = 0; i < c; i++) {
          T distance = 0;
          for (int j = 0; j < inner_dim; j++) {
            distance += std::pow(means[i * inner_dim + j] - tmp[j], minimam);
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

// 収束したりしなかったり
// // 112.374572754
// // 112.374725342
// const T weight_beta = 112.374572754, weight_eps = 0.01;
// opt::RMSProp optimizer(0.0001, 0.9);
// int batch = 3;
// int class_n = 2, dim = 200, inner_dim = dim;
// int M = 60;
// int test_N = 20;

int main() {
  const T weight_beta = 20, weight_eps = 0.01;
  opt::RMSProp optimizer(0.0001, 0.9);
  int batch = 15;
  int class_n = 2, dim = 200, inner_dim = 2;
  int M = 60;
  int test_N = batch;

  Logger log("../../log/rnn_classify.csv");

  DataMaker maker(dim, inner_dim, class_n, M, 0.01);

  std::vector<std::vector<T>> test_X;  //(M * test_N);
  std::vector<int> test_Y;             //(M * test_N);
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < test_N; n++) {
      std::vector<T> res(dim);
      auto label = maker.gen(res.data(), m);

      test_X.push_back(res);
      test_Y.push_back(label);
    }
  }

  auto rnn = std::make_unique<custom::RNN>(batch, dim, 10);
  // auto cnn = std::make_unique<linear::Layer>(batch, dim, dim);
  // auto act = std::make_unique<act::Layer<c1func::Tanh>>(*cnn);
  auto out = std::make_unique<linear::Layer>(*rnn, class_n);
  auto soft = std::make_unique<act::SoftMax>(*out);

  // auto out = std::make_unique<linear::Layer>(batch, dim, class_n);
  // auto soft = std::make_unique<act::SoftMax>(*out);

  auto rnn_state = rnn->state.get();
  auto rnn_state_n = rnn->b_n * rnn->n;
  auto reset_state = [=]() { std::fill_n(rnn_state, rnn_state_n, T(0)); };
  {
    std::mt19937 engine(42);
    std::normal_distribution<> dist(0.0, weight_beta / std::sqrt(rnn->n));

    std::fill_n(rnn->weight.get(), rnn->weight_size(), T(0));
    for (int i = 0; i < rnn->n; i++)
      rnn->weight[i * rnn->n + i] = (1 - weight_eps) * T(1);
    for (int i = 0; i < rnn->weight_size(); i++)
      rnn->weight[i] += weight_eps * dist(engine);

    std::fill_n(rnn->bias.get(), rnn->bias_size(), T(0));
  }
  out->xivier(44);

  // cnn->xivier(45);

  Model<decltype(optimizer)> model(optimizer);
  model.add_layer(std::move(rnn));
  // model.add_layer(std::move(cnn));
  // model.add_layer(std::move(act));
  model.add_layer(std::move(out));
  model.add_layer(std::move(soft));

  CrossEntropy metrics(batch, class_n);

  auto evaluate = [&]() {
    T metric_score = 0;
    int correct = 0, N = 0;
    for (int e = 0; e < test_X.size() / batch; e++) {
      int batch_Y[batch];
      for (int b = 0; b < batch; b++) {
        std::copy_n(test_X[e * batch + b].data(), dim, &(model.x())[b * dim]);
        batch_Y[b] = test_Y[e * batch + b];
      }

      model.forward();

      metric_score += metrics(model.y(), batch_Y);
      for (int b = 0; b < batch; b++) {
        int y_act = argmax_n(&model.y()[b * class_n], class_n);
        int y_true = batch_Y[b];

        if (y_act == y_true) correct++;
        N++;
      }
    }
    metric_score /= test_X.size();

    printf("test_score = %lf, test_acc = %lf\n", metric_score, (T)correct / N);
    return std::tuple<T, T>(metric_score, (T)correct / N);
  };
  auto accuracy = [&](int *batch_Y) {
    int correct = 0, N = 0;
    for (int b = 0; b < batch; b++) {
      int y_act = argmax_n(&model.y()[b * class_n], class_n);
      int y_true = batch_Y[b];

      if (y_act == y_true) correct++;
      N++;
    }
    return (T)correct / N;
  };

  int epochs = 20000;
  int eval = 100;
  for (int e = 0; e < epochs; e++) {
    int batch_Y[batch];
    for (int b = 0; b < batch; b++) {
      batch_Y[b] = maker.random_gen(&(model.x())[b * dim]);
    }

    // reset_state();
    model.forward();
    if (e % eval == 0) {
      T score = metrics(model.y(), batch_Y);
      T acc = accuracy(batch_Y);
      printf("%d: batch_score = %lf, batch_acc = %lf, ", e, score, acc);
      auto res = evaluate();

      log.print("%d,%f,%f,%f\n", e, score, std::get<0>(res), std::get<1>(res));
    }
    metrics.d(model.y(), batch_Y, model.y());
    model.backward();
    model.update();
  }
}