
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

class DataProc {
 public:
  DataProc(int dim, int inner_dim, int class_n, int cluster_n, T noise_scale,
           int seed)
      : dim(dim),
        inner_dim(inner_dim),
        class_n(class_n),
        cluster_n(cluster_n),
        engine(std::mt19937(seed)),
        noise_scale(noise_scale),
        noise(std::normal_distribution<>(0.0, noise_scale)),
        rand(std::uniform_int_distribution<>(0, cluster_n - 1)) {
    means = std::make_unique<T[]>(cluster_n * inner_dim);
    labels = std::make_unique<int[]>(cluster_n);
    w = std::make_unique<T[]>(dim * inner_dim);

    init_w();
    init_means();
  }

  int random_gen(T *res) {
    int c = rand(engine);  // std::rand() % cluster_n;
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
  std::uniform_int_distribution<int> rand;
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
        noise_scale * 10;  // 標準偏差の5倍の距離は保つ　外に出る確率が1e-7

    T tmp[inner_dim];
    const T L = 4.0;
    std::uniform_real_distribution<> unif(-L / 2, L / 2);
    for (int c = 0; c < cluster_n; c++) {
      bool flag = true;
      while (flag) {
        // uniform sample from inner_dim's hypercube
        for (auto &x : tmp) x = unif(engine);

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

int main() {
  const T weight_beta = 250, weight_eps = 0.01;
  const T out_ro = 0.3;
  const T opt_lr = 0.001, opt_beta = 0.99;
  const int batch = 10;
  const int class_n = 2, dim = 200, inner_dim = 2;
  const T noise_scale = 0.02;
  const int M = 60;
  const int test_N = batch;
  const int seed = 42;
  const int patience = 5;

  Logger log("../../log/rnn_classify.csv");
  std::mt19937 engine(seed);

  DataProc proc(dim, inner_dim, class_n, M, noise_scale, engine());

  std::vector<std::vector<T>> test_X;  //(M * test_N);
  std::vector<int> test_Y;             //(M * test_N);
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < test_N; n++) {
      std::vector<T> res(dim);
      auto label = proc.gen(res.data(), m);

      test_X.push_back(res);
      test_Y.push_back(label);
    }
  }

  auto rnn = std::make_unique<custom::RNN>(batch, dim, 10);
  auto out = std::make_unique<linear::Layer>(*rnn, class_n);
  auto soft = std::make_unique<act::SoftMax>(*out);

  {
    std::normal_distribution<> dist(0.0, weight_beta / std::sqrt(rnn->n));

    std::fill_n(rnn->weight.get(), rnn->weight_size(), T(0));
    for (int i = 0; i < rnn->n; i++)
      rnn->weight[i * rnn->n + i] = (1 - weight_eps) * T(1);
    for (int i = 0; i < rnn->weight_size(); i++)
      rnn->weight[i] += weight_eps * dist(engine);

    std::fill_n(rnn->bias.get(), rnn->bias_size(), T(0));
  }
  out->normal(out_ro, engine());

  opt::RMSProp optimizer(opt_lr, opt_beta);
  Model<decltype(optimizer)> model(optimizer);
  model.add_layer(std::move(rnn));
  model.add_layer(std::move(out));
  model.add_layer(std::move(soft));

  CrossEntropy metrics(batch, class_n);

  auto eval_test = [&]() {
    T metric_score = 0;
    int correct = 0, N = 0;
    for (int e = 0; e < test_X.size() / batch; e++) {
      int batch_Y[batch];
      for (int b = 0; b < batch; b++) {
        std::copy_n(test_X[e * batch + b].data(), dim, &(model.x())[b * dim]);
        batch_Y[b] = test_Y[e * batch + b];
      }
      model.reset();
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

  ReduceOnPleatou<decltype(optimizer)> scheduler(model, patience, 0.5);

  const int epochs = 1000000;
  const int eval = 100;
  for (int e = 0; e < epochs; e++) {
    int batch_Y[batch];
    for (int b = 0; b < batch; b++) {
      batch_Y[b] = proc.random_gen(&(model.x())[b * dim]);
    }
    model.reset();
    model.forward();
    if (e % eval == 0) {
      T batch_score = metrics(model.y(), batch_Y);
      T batch_acc = accuracy(batch_Y);
      T test_score, test_acc;
      std::tie(test_score, test_acc) = eval_test();

      printf(
          "%d: lr = %lf, batch_score = %lf, batch_acc = %lf, test_score = %lf, "
          "test_acc = %lf\n",
          e, model.opts[0]->alpha, batch_score, batch_acc, test_score,
          test_acc);
      log.print("%d,%f,%f,%f\n", e, batch_score, test_score, test_acc);

      if (scheduler.step(test_score)) {
        printf("learning late halfed\n");
      }
    }
    metrics.d(model.y(), batch_Y, model.y());
    model.backward();
    model.update();
  }
}