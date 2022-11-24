#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include "common.h"
#include "mynn.h"
#include "mynn_custom.h"
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
            distance += std::pow(means[i * inner_dim + j] - tmp[j], 2.0);
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

void experiment(T weight_beta, int inner_dim, int patience, int model_seed) {
  const T weight_eps = 0.01;
  const T out_ro = 0.3;
  const T opt_lr = 0.001, opt_beta = 0.99;
  const int batch = 10;
  const int class_n = 2, dim = 200;
  const T noise_scale = 0.02;
  const int M = 60;
  const int test_N = batch;
  const int rnn_t = 10;
  const int data_seed = 42;

  std::mt19937 engine(model_seed);

  DataProc proc(dim, inner_dim, class_n, M, noise_scale, data_seed);

  std::vector<std::vector<T>> test_X;
  std::vector<int> test_Y;
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < test_N; n++) {
      std::vector<T> res(dim);
      auto label = proc.gen(res.data(), m);

      test_X.push_back(res);
      test_Y.push_back(label);
    }
  }

  resource::Param rnn_w(dim * dim), rnn_b(dim);
  resource::Param out_w(dim * class_n), out_b(class_n);

  model::Composite model;
  {
    auto rnn = std::make_unique<model::RNN>(rnn_t, rnn_w, rnn_b);
    auto out = std::make_unique<model::Affine>(*rnn, out_w, out_b);
    auto soft = std::make_unique<model::SoftMax>(*out);

    {
      std::normal_distribution<> dist(0.0, weight_beta / std::sqrt(rnn->n));

      std::fill_n(rnn->w.v(), rnn->w.size, T(0));
      for (int i = 0; i < rnn->n; i++)
        rnn->w._v[i * rnn->n + i] = (1 - weight_eps) * T(1);
      for (int i = 0; i < rnn->w.size; i++)
        rnn->w._v[i] += weight_eps * dist(engine);

      std::fill_n(rnn->b.v(), rnn->b.size, T(0));
    }
    affine_normal(*out, out_ro, engine());

    model.add(down_cast<model::ModelBase>(std::move(rnn)));
    model.add(down_cast<model::ModelBase>(std::move(out)));
    model.add(down_cast<model::ModelBase>(std::move(soft)));
  }

  {  // train
    policy::RMSProp optimizer(opt_lr, opt_beta);
    optimizer.add_target(rnn_w, rnn_b, out_w, out_b);

    loss_func::CrossEntropy metric(class_n);
    lr::ReduceOnPleatou<decltype(optimizer)> scheduler(optimizer, patience,
                                                       0.5);
    T score;

    auto nn = model.create(batch);
    auto met = metric.create(batch);

    auto eval_test = [&]() {
      T metric_score = 0;
      Acc accuracy;
      for (int e = 0; e < test_X.size() / batch; e++) {
        for (int b = 0; b < batch; b++) {
          std::copy_n(test_X[e * batch + b].data(), dim,
                      &(nn->input())[b * dim]);
          met->_correct[b] = test_Y[e * batch + b];
        }
        nn->reset();
        met->reset();

        nn->forward(met->input());
        met->forward(&score);

        metric_score += score;
        for (int b = 0; b < batch; b++) {
          int y_act = argmax_n(&met->input()[b * class_n], class_n);
          int y_true = met->_correct[b];

          accuracy.step(y_act == y_true);
        }
      }
      metric_score /= test_X.size() / batch;

      return std::tuple<T, T>(metric_score, accuracy.result());
    };

    Logger log("../../log/rnn_classify.csv");

    const int epochs = 96000;
    const int eval = 100;
    for (int e = 0; e < epochs; e++) {
      int batch_Y[batch];
      for (int b = 0; b < batch; b++) {
        met->_correct[b] = proc.random_gen(&(nn->input())[b * dim]);
      }
      nn->reset();
      met->reset();

      nn->forward(met->input());
      met->forward(&score);

      if (e % eval == 0) {
        T batch_score = score;
        T test_score, test_acc;
        std::tie(test_score, test_acc) = eval_test();

        printf(
            "%d: lr = %lf, batch_score = %lf, test_score = "
            "%lf, "
            "test_acc = %lf\n",
            e, scheduler.current_lr(), batch_score, test_score, test_acc);
        log.print("%d,%f,%f,%f\n", e, batch_score, test_score, test_acc);

        if (scheduler.current_lr() < 1e-7 || test_score < 1e-7) break;
        if (scheduler.step(test_score)) {
          printf("learning late halfed\n");
        }
      }
      met->backward();
      nn->backward(met->input());

      optimizer.update();
    }
  }

  {
    const int span = rnn_t * 5;
    std::vector<std::vector<std::vector<T>>> snapshots(
        span, std::vector<std::vector<T>>());

    {
      auto _affine = std::make_unique<model::Affine>(rnn_w, rnn_b);
      auto _tanh = std::make_unique<model::Act<c1func::Tanh>>(*_affine);

      auto affine = _affine->create(batch);
      auto tanh = _tanh->create(batch);
      std::vector<T> input(batch * dim);

      auto reset = [&]() { std::fill_n(affine->input(), batch * dim, T(0)); };
      auto first_forward = [&]() {
        affine->forward(tanh->input());
        for (int i = 0; i < batch * dim; i++) {
          tanh->input()[i] += input[i];
        }
        tanh->forward(affine->input());
      };
      auto forward = [&]() {
        affine->forward(tanh->input());
        tanh->forward(affine->input());
      };

      for (int e = 0; e < test_X.size() / batch; e++) {
        for (int b = 0; b < batch; b++) {
          std::copy_n(test_X[e * batch + b].data(), dim, &input[b * dim]);
        }
        reset();
        for (int t = 0; t < span; t++) {
          (t == 0) ? first_forward() : forward();

          for (int b = 0; b < batch; b++) {
            std::vector<T> state(dim);
            std::copy_n(&(affine->input())[b * dim], dim, state.data());
            snapshots[t].emplace_back(state);
          }
        }
      }
    }

    char dir[128];
    std::sprintf(dir, "../../log/%d_%d_%d_%d", (int)weight_beta, inner_dim,
                 patience, model_seed);
    mkdir(dir, 0777);
    for (int t = 0; t < span; t++) {
      char path[256];

      std::sprintf(path, "%s/%d.csv", dir, t + 1);
      Logger log(path);

      for (auto &v : snapshots[t]) {
        log.print("%f", v[0]);
        for (int i = 1; i < v.size(); i++) log.print(",%f", v[i]);
        log.print("\n");
      }
    }
    {
      char path[256];
      std::sprintf(path, "%s/labels.csv", dir);
      Logger log(path);
      for (auto &label : test_Y) {
        log.print("%d\n", label);
      }
    }
  }
}

int main() {
  std::mt19937 engine(42);
  const int n = 10;

  std::vector<int> seeds;
  for (int i = 0; i < n; i++) seeds.emplace_back(engine());

  for (auto seed : seeds) experiment(1, 2, 5, seed);
  for (auto seed : seeds) experiment(20, 2, 5, seed);
  for (auto seed : seeds) experiment(100, 2, 5, seed);
  for (auto seed : seeds) experiment(250, 2, 5, seed);
}