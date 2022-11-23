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
  const int l2_n = 100;
  const int l3_n = y_n;

  resource::Param w0(l0_n * l1_n), b0(l1_n);
  resource::Param w1(l1_n * l2_n), b1(l2_n);
  resource::Param w2(l2_n * l3_n), b2(l3_n);

  model::Composite model;
  {
    auto lay0 = std::make_unique<model::Affine>(w0, b0);
    auto act0 = std::make_unique<model::Act<c1func::Tanh>>(*lay0);
    auto lay1 = std::make_unique<model::Affine>(w1, b1);
    auto act1 = std::make_unique<model::Act<c1func::Tanh>>(*lay1);
    auto lay2 = std::make_unique<model::Affine>(w2, b2);
    auto act2 = std::make_unique<model::SoftMax>(*lay2);

    affine_normal(*lay0, 1.0, 42);
    affine_normal(*lay1, 1.0, 43);
    affine_normal(*lay2, 1.0, 44);

    model.add(down_cast<model::ModelBase>(std::move(lay0)));
    model.add(down_cast<model::ModelBase>(std::move(act0)));
    model.add(down_cast<model::ModelBase>(std::move(lay1)));
    model.add(down_cast<model::ModelBase>(std::move(act1)));
    model.add(down_cast<model::ModelBase>(std::move(lay2)));
    model.add(down_cast<model::ModelBase>(std::move(act2)));
  }
  {  // train
    auto train_X = mnist.train_data();
    auto train_Y = mnist.train_label();

    policy::RMSProp optimizer(0.001, 0.9);
    optimizer.add_target(w0, b0, w1, b1, w2, b2);

    loss_func::CrossEntropy metrics(y_n);

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
    int death_count = 0;
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
        if (death_count >= 10)
          break;
        else {
          if (test_score < min_test_score) {
            min_test_score = test_score;

            death_count = 0;
          } else {
            death_count++;
          }
        }
      }

      met->backward();
      nn->backward(met->input());
      optimizer.update();
    }
  }
}
