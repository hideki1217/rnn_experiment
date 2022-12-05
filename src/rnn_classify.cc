#include "rnn_classify.h"

#include <cblas.h>
#include <sys/stat.h>

#include <cassert>
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
  const int test_N = batch * 3;
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

  std::vector<std::tuple<int, T, T, T, T>> learning_log;
  auto report_learning_log = [&](int samples, T lr, T batch_loss, T test_loss,
                                 T test_acc) {
    learning_log.emplace_back(
        std::make_tuple(samples, lr, batch_loss, test_loss, test_acc));
  };

  {  // train
    policy::RMSProp optimizer(opt_lr, opt_beta);
    optimizer.add_target(rnn_w, rnn_b, out_w, out_b);

    loss_func::CrossEntropy metric(class_n);
    lr::ReduceOnPleatou<decltype(optimizer)> scheduler(optimizer, patience,
                                                       0.5);
    T loss;

    auto nn = model.create(batch);
    auto met = metric.create(batch);

    auto eval_test = [&]() {
      T total_loss = 0;
      Acc accuracy;
      const int epochs = test_X.size() / batch;
      for (int e = 0; e < epochs; e++) {
        for (int b = 0; b < batch; b++) {
          std::copy_n(test_X[e * batch + b].data(), dim,
                      &(nn->input())[b * dim]);
          met->_correct[b] = test_Y[e * batch + b];
        }
        nn->reset();
        met->reset();

        nn->forward(met->input());
        met->forward(&loss);

        total_loss += loss;
        for (int b = 0; b < batch; b++) {
          int y_act = argmax_n(&met->input()[b * class_n], class_n);
          int y_true = met->_correct[b];

          accuracy.step(y_act == y_true);
        }
      }
      total_loss /= epochs;

      return std::tuple<T, T>(total_loss, accuracy.result());
    };

    const int max_epochs = 60;
    std::vector<int> spectoram({0, 1, 4, 14, 29, 59});
    for (int e = 0; e < max_epochs; e++) {
      T total_loss = 0;
      for (int i = 0; i < iteration; i++) {
        for (int b = 0; b < batch; b++) {
          met->_correct[b] = proc.random_gen(&(nn->input())[b * dim]);
        }

        nn->reset();
        met->reset();

        nn->forward(met->input());
        met->forward(&loss);

        total_loss += loss;

        met->backward();
        nn->backward(met->input());

        optimizer.update();
      }
      T batch_loss = total_loss / iteration;
      scheduler.step(batch_loss);

      {  // report current state
        T current_lr = scheduler.current_lr();
        T test_loss, test_acc;
        std::tie(test_loss, test_acc) = eval_test();
        report_learning_log((e + 1) * iteration * batch, current_lr, batch_loss,
                            test_loss, test_acc);

        printf(
            "%d(%d): lr = %lf, batch_loss = %lf, test_loss = "
            "%lf, "
            "test_acc = %lf\n",
            e, e * iteration * batch, current_lr, batch_loss, test_loss,
            test_acc);
      }
      if (std::find(spectoram.begin(), spectoram.end(), e) != spectoram.end()) {
        calc_spectoram(e);
      }
    }
  }

  {  // save learning log
    auto log_path = std::string(savedir) + "/learning_log.csv";
    Logger log(log_path);
    log.print("epoch,lr,batch_loss,test_loss,teset_acc\n");
    for (auto x : learning_log) {
      int samples;
      T lr, batch_loss, test_loss, test_acc;
      std::tie(samples, lr, batch_loss, test_loss, test_acc) = x;

      log.print("%d,%f,%f,%f,%f\n", samples, lr, batch_loss, test_loss,
                test_acc);
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

int main(int argc, char *argv[]) {
  openblas_set_num_threads(8);

  std::mt19937 engine(42);
  const int n = 30;

  std::vector<int> seeds;
  for (int i = 0; i < n; i++) seeds.emplace_back(engine());

  assert(argc == 4);
  auto g_radius = std::stoi(argv[1]);
  auto inner_dim = std::stoi(argv[2]);
  auto patience = std::stoi(argv[3]);

  for (auto seed : seeds) experiment(g_radius, inner_dim, patience, seed);
}