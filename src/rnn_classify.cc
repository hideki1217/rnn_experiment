#include "rnn_classify.h"

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
#include "ryapunov.h"

using namespace mynn;

void experiment(T weight_beta, int inner_dim, int patience, int model_seed) {
  char savedir[128];
  std::sprintf(savedir, "../../log/%d_%d_%d_%d", (int)weight_beta, inner_dim,
               patience, model_seed);
  {  // もしすでに走らせたことのあるパラメータならもう走らせない。
    struct stat st;
    if (stat(savedir, &st) == 0) {
      return;
    }
    mkdir(savedir, 0777);
  }
  std::printf("%d_%d_%d_%d\n", (int)weight_beta, inner_dim, patience,
              model_seed);

  const T weight_eps = 0.01;
  const T out_ro = 0.3;
  const T opt_lr = 0.001, opt_beta = 0.99;
  const int batch = 10;
  const int class_n = 2, dim = 200;
  const T noise_scale = 0.02;
  const int M = 60;
  const int test_N = batch;
  const int rnn_t = 10;
  const int iteration = 800;
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
    auto rnn = std::make_unique<model::RNN<c1func::Tanh>>(rnn_t, rnn_w, rnn_b);
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

  std::vector<std::tuple<int, std::vector<T>>> spectorams;
  auto calc_spectoram = [&](int epoch) {
    ryap::RNN rnn(dim);
    std::copy_n(rnn_w.v(), dim * dim, rnn.weight.begin());
    std::copy_n(rnn_b.v(), dim, rnn.bias.begin());

    spectorams.emplace_back(
        std::make_tuple(epoch, ryap::spectoram(std::move(rnn))));
  };

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

    auto log_path = std::string(savedir) + "/learning_log.csv";
    Logger log(log_path);
    log.print("epoch,lr,batch_score,test_score,teset_acc\n");

    const int max_epochs = 50;
    const int spectoram = 5;
    for (int e = 0; e < max_epochs; e++) {
      for (int i = 0; i < iteration; i++) {
        for (int b = 0; b < batch; b++) {
          met->_correct[b] = proc.random_gen(&(nn->input())[b * dim]);
        }

        nn->reset();
        met->reset();

        nn->forward(met->input());
        met->forward(&score);

        met->backward();
        nn->backward(met->input());

        optimizer.update();
      }

      T batch_score = score;
      T test_score, test_acc;
      std::tie(test_score, test_acc) = eval_test();
      printf(
          "%d(%d): lr = %lf, batch_score = %lf, test_score = "
          "%lf, "
          "test_acc = %lf\n",
          e, e * iteration * batch, scheduler.current_lr(), batch_score, test_score, test_acc);
      log.print("%d,%f,%f,%f,%f\n", e, scheduler.current_lr(), batch_score,
                test_score, test_acc);

      if (scheduler.step(test_score)) {
        printf("learning late halfed\n");
      }
      if (scheduler.current_lr() < 1e-7 || test_score < 1e-7) {
        calc_spectoram(e);
        break;
      }
      if (e % spectoram == 0) {
        calc_spectoram(e);
      }
    }
  }

  {
    auto path = std::string(savedir) + "/spectoram.csv";
    Logger log(path);
    for (auto &res : spectorams) {
      int epoch = std::get<0>(res);
      std::vector<T> &spectoram = std::get<1>(res);

      log.print("%d", epoch);
      for (auto &s : spectoram) {
        log.print(",%f", s);
      }
      log.print("\n");
    }
  }

  {
    const int span = rnn_t * 5;
    std::vector<std::vector<std::vector<T>>> snapshots(
        span, std::vector<std::vector<T>>());

    {
      auto rnn = model.models[0]->create_for_eval(batch);
      std::vector<T> y(batch * dim);

      for (int e = 0; e < test_X.size() / batch; e++) {
        rnn->reset();

        for (int t = 0; t < span; t++) {
          if (t == 0) {
            for (int b = 0; b < batch; b++) {
              std::copy_n(test_X[e * batch + b].data(), dim,
                          &(rnn->input())[b * dim]);
            }
          } else {
            std::fill_n(rnn->input(), batch * dim, T(0));
          }

          rnn->forward(y.data());

          for (int b = 0; b < batch; b++) {
            std::vector<T> state(dim);
            std::copy_n(&y[b * dim], dim, state.data());
            snapshots[t].emplace_back(state);
          }
        }
      }
    }

    for (int t = 0; t < span; t++) {
      char path[256];

      std::sprintf(path, "%s/%d.csv", savedir, t + 1);
      Logger log(path);

      for (auto &v : snapshots[t]) {
        log.print("%f", v[0]);
        for (int i = 1; i < v.size(); i++) log.print(",%f", v[i]);
        log.print("\n");
      }
    }
    {
      char path[256];
      std::sprintf(path, "%s/labels.csv", savedir);
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

  for (auto inner_dim : {2, 200}) {
    for (auto beta : {1, 20, 100, 250}) {
      for (auto seed : seeds) experiment(beta, inner_dim, 5, seed);
    }
  }
}