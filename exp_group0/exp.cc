#include <cblas.h>
#include <sys/stat.h>

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include "common.h"
#include "mynn.h"
#include "mynn_custom.h"
#include "mynn_util.h"
#include "rnn_classify.h"

using namespace mynn;

#define PRINT_LEARNING_LOG 0

#define SAVE_LEARNING_LOG 1
#define SAVE_MODEL_SNAPSHOTS 1
#define SAVE_DYNAMICS_SNAPSHOTS 1

#define SAVE_ALL \
  SAVE_LEARNING_LOG &SAVE_MODEL_SNAPSHOTS &SAVE_DYNAMICS_SNAPSHOTS

void experiment(T weight_beta, int inner_dim, int patience, int model_seed) {
  char savedir[128];
  std::sprintf(savedir, "../../exp_group0/" EXP_NAME "/log/%d_%d_%d_%d",
               (int)weight_beta, inner_dim, patience, model_seed);
#if SAVE_ALL
  {  // もしすでに走らせたことのあるパラメータならもう走らせない。
    struct stat st;
    if (stat(savedir, &st) == 0) {
      return;
    }
  }
#endif
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

#if EXP_ID == 0
  DataProc proc(dim, inner_dim, class_n, M, noise_scale, data_seed);
#else
  // 選択行列
  auto w_in = std::make_unique<T[]>(dim * inner_dim);
  std::fill_n(w_in.get(), dim * inner_dim, T(0));
  for (int i = 0; i < std::min(dim, inner_dim); i++) {
    w_in[i * inner_dim + i] = T(1);
  }
  LinEmbed proc(dim, std::move(w_in), inner_dim, class_n, M, noise_scale,
                data_seed);
#endif

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

  std::vector<std::tuple<int, std::vector<T>>> model_snapshots;
  auto save_model = [&](int epoch) {
    std::vector<T> snapshot((dim + 1) * dim);
    std::copy_n(rnn_w.v(), dim * dim, &snapshot[0]);
    std::copy_n(rnn_b.v(), dim, &snapshot[dim * dim]);
    model_snapshots.emplace_back(std::make_tuple(epoch, std::move(snapshot)));
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
    std::vector<int> save_epoch_list({0, 1, 2, 5, 10, 15, 20, 25, 30, 45, 60});
    for (int epoch = 0;;) {
      if (std::find(save_epoch_list.begin(), save_epoch_list.end(), epoch) !=
          save_epoch_list.end()) {
        save_model(epoch);
      }
      if (epoch >= max_epochs) break;

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
      epoch++;

      {  // report current state
        T current_lr = scheduler.current_lr();
        T test_loss, test_acc;
        std::tie(test_loss, test_acc) = eval_test();
        report_learning_log(epoch * iteration * batch, current_lr, batch_loss,
                            test_loss, test_acc);

#if PRINT_LEARNING_LOG
        printf(
            "%d(%d): lr = %lf, batch_loss = %lf, test_loss = "
            "%lf, "
            "test_acc = %lf\n",
            epoch, epoch * iteration * batch, current_lr, batch_loss, test_loss,
            test_acc);
#endif
      }
    }
  }

  // start save logs
  mkdir(savedir, 0777);

#if SAVE_LEARNING_LOG
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
#endif

#if SAVE_MODEL_SNAPSHOTS
  {  // save model snapshot
    for (auto &x : model_snapshots) {
      int epoch;
      std::vector<T> snapshot;
      std::tie(epoch, snapshot) = x;

      auto path =
          std::string(savedir) + "/model_" + std::to_string(epoch) + ".csv";
      Logger log(path);
      for (int i = 0; i < dim + 1; i++) {
        log.print("%f", snapshot[i * dim]);
        for (int j = 1; j < dim; j++) {
          log.print(",%f", snapshot[i * dim + j]);
        }
        log.print("\n");
      }
    }
  }
#endif

#if SAVE_DYNAMICS_SNAPSHOTS
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
#endif
}

int main(int argc, char *argv[]) {
  openblas_set_num_threads(8);

  std::mt19937 engine(42);
  const int n = 10;

  std::vector<int> seeds;
  for (int i = 0; i < n; i++) seeds.emplace_back(engine());

  assert(argc == 4);
  auto g_radius = std::stoi(argv[1]);
  auto inner_dim = std::stoi(argv[2]);
  auto patience = std::stoi(argv[3]);

  auto start = std::chrono::system_clock::now();
  for (auto seed : seeds) experiment(g_radius, inner_dim, patience, seed);
  auto time = std::chrono::system_clock::now() - start;
  double time_m =
      std::chrono::duration_cast<std::chrono::minutes>(time).count();
  printf("PROCESS %d_%d_%d END: time(minutes) = %f\n", g_radius, inner_dim,
         patience, time_m);
}