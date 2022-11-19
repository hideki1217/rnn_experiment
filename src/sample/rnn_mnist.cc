#include <cmath>
#include <vector>

#include "../common.h"
#include "../dataset.h"
#include "../mynn.h"
#include "../mynn_util.h"

using namespace mynn;

int main() {
  const int batch = 20, x_n = 28 * 28, y_n = 10;

  auto rnn = std::make_unique<custom::RNN>(batch, x_n, 3);
  auto lay1 = std::make_unique<linear::Layer>(*rnn, y_n);
  auto act1 = std::make_unique<act::SoftMax>(*lay1);

  // lay0->xivier(42);
  lay1->xivier(43);

  Model<opt::Adam> model(opt::Adam(0.001, 0.9, 0.999));
  // model.add_layer(std::move(lay0));
  model.add_layer(std::move(rnn));
  model.add_layer(std::move(lay1));
  model.add_layer(std::move(act1));

  CrossEntropy metrics(batch, y_n);

  dataset::Mnist mnist;
  auto test_X = mnist.test_data();
  auto test_Y = mnist.test_label();

  auto evaluate = [&]() {
    T metric_score = 0;
    int correct = 0, N = 0;
    for (int e = 0; e < test_X.size() / batch; e++) {
      int batch_Y[batch];
      for (int b = 0; b < batch; b++) {
        std::copy_n(test_X[e * batch + b].data(), x_n, &(model.x())[b * x_n]);
        batch_Y[b] = test_Y[e * batch + b];
      }
      model.reset();
      model.forward();

      metric_score += metrics(model.y(), batch_Y);
      for (int b = 0; b < batch; b++) {
        int y_act = argmax_n(&model.y()[b * y_n], y_n);
        int y_true = batch_Y[b];

        if (y_act == y_true) correct++;
        N++;
      }
    }

    printf("metric = %lf, accuracy = %lf\n", metric_score, (T)correct / N);
    return metric_score;
  };
  {
    auto train_X = mnist.train_data();
    auto train_Y = mnist.train_label();

    int max_epochs = 1000000;
    std::vector<int> indexs(batch * max_epochs);
    for (int i = 0; i < indexs.size(); i++) {
      indexs[i] = i % test_X.size();
    }
    std::random_shuffle(indexs.begin(), indexs.end());

    T min_score = 1e6;
    int evaluate_count = 200;
    int death_count = 0;
    for (int e = 0; e < max_epochs; e++) {
      int batch_Y[batch];
      for (int b = 0; b < batch; b++) {
        std::copy_n(train_X[indexs[e * batch + b]].data(), x_n,
                    &(model.x())[b * x_n]);
        batch_Y[b] = train_Y[indexs[e * batch + b]];
      }
      model.reset();
      model.forward();

      if (e % evaluate_count == 0) {
        printf("%d: ", e);
        auto score = evaluate();
        if (death_count >= 5)
          break;
        else {
          if (score < min_score) {
            min_score = score;

            death_count = 0;
          } else {
            death_count++;
          }
        }
      }

      metrics.d(model.y(), batch_Y, model.y());
      model.backward();
      model.update();
    }
  }
  evaluate();
}