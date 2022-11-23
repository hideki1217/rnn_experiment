#include <random>

#include "../common.h"
#include "../dataset.h"
#include "../mynn.h"
#include "../mynn_custom.h"
#include "../mynn_util.h"
using namespace mynn;

int main() {
  dataset::Mnist mnist;
  auto test_X = mnist.test_data();
  auto test_Y = mnist.test_label();

  const int batch = 100, x_n = 28 * 28, y_n = 10;
  const int l0_n = x_n;
  const int l1_n = 100;
  const int l2_n = y_n;

  resource::Param w0(l0_n * l1_n), b0(l1_n);
  resource::Param rnn_w(l1_n * l1_n), rnn_b(l1_n);
  resource::Param w1(l1_n * l2_n), b1(l2_n);

  model::Composite model;
  {
    auto lay0 = std::make_unique<model::Affine>(w0, b0);
    auto rnn = std::make_unique<model::RNN>(5, rnn_w, rnn_b);
    auto lay1 = std::make_unique<model::Affine>(w1, b1);
    auto act1 = std::make_unique<model::SoftMax>(*lay1);

    affine_normal(*lay0, 1.0, 42);
    affine_normal(rnn->w, rnn->b, 1.0, 43);
    affine_normal(*lay1, 1.0, 44);

    model.add(down_cast<model::ModelBase>(std::move(lay0)));
    model.add(down_cast<model::ModelBase>(std::move(rnn)));
    model.add(down_cast<model::ModelBase>(std::move(lay1)));
    model.add(down_cast<model::ModelBase>(std::move(act1)));
  }
  {  // train
    auto train_X = mnist.train_data();
    auto train_Y = mnist.train_label();

    policy::Momentum optimizer(0.001, 0.9);
    optimizer.add_target(w0, b0, rnn_w, rnn_b, w1, b1);

    loss_func::CrossEntropy metrics(y_n);
    lr::ReduceOnPleatou<decltype(optimizer)> scheduler(optimizer, 5, 0.5);

    auto nn = model.create(batch);
    auto met = metrics.create(batch);

    T score;

    auto evaluate = [&]() {
      T metric_score = 0;
      Acc accuracy;
      for (int e = 0; e < test_X.size() / batch; e++) {
        for (int b = 0; b < batch; b++) {
          std::copy_n(test_X[e * batch + b].data(), x_n,
                      &(nn->input())[b * x_n]);
          met->_correct[b] = test_Y[e * batch + b];
        }

        nn->reset();
        met->reset();

        nn->forward(met->input());
        met->forward(&score);

        metric_score += score;
        for (int b = 0; b < batch; b++) {
          int y_act = argmax_n(&(met->input())[b * y_n], y_n);
          int y_true = met->_correct[b];

          accuracy.step(y_act == y_true);
        }
      }
      metric_score /= test_X.size() / batch;

      printf("metric = %lf, accuracy = %lf\n", metric_score, accuracy.result());
      return metric_score;
    };

    int max_epochs = 1000000;
    std::vector<int> indexs(batch * max_epochs);
    for (int i = 0; i < indexs.size(); i++) {
      indexs[i] = i % test_X.size();
    }
    std::random_shuffle(indexs.begin(), indexs.end());

    T min_test_score = 1e6;
    int evaluate_count = 200;
    for (int e = 0; e < max_epochs; e++) {
      for (int b = 0; b < batch; b++) {
        std::copy_n(train_X[indexs[e * batch + b]].data(), x_n,
                    &(nn->input())[b * x_n]);
        met->_correct[b] = train_Y[indexs[e * batch + b]];
      }
      nn->reset();
      met->reset();

      nn->forward(met->input());
      met->forward(&score);
      if (e % evaluate_count == 0) {
        printf("%d: batch_score = %lf, ", e, score);
        auto test_score = evaluate();
        scheduler.step(test_score);
        if (scheduler.current_lr() < 1e-5) break;
      }

      met->backward();
      nn->backward(met->input());
      optimizer.update();
    }
  }
}
