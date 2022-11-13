#include <memory>
#include <random>
#include <vector>

#include "mynn.h"

using namespace mynn;

class MSE {
 public:
  MSE(int n) : n(n) {}
  T operator()(const T* y_act, const T* y_true) {
    T res = 0, _n = 1.0 / n;
    for (int i = 0; i < n; i++) {
      res += std::pow(y_act[i] - y_true[i], 2) * _n;
    }
    return res / 2.0;
  }

  void d(const T* y_act, const T* y_true, T* dy) {
    T _n = 1.0 / n;
    for (int i = 0; i < n; i++) {
      dy[i] = (y_act[i] - y_true[i]) * _n;
    }
  }
  int n;
};

int nn_sample() {
  const int batch = 100, x_n = 1, y_n = 1;

  auto lay0 = std::make_unique<linear::Layer>(batch, x_n, 100);
  auto act0 = std::make_unique<act::Layer<c1func::Tanh>>(*lay0);
  auto lay1 = std::make_unique<linear::Layer>(*act0, 100);
  auto act1 = std::make_unique<act::Layer<c1func::Tanh>>(*lay1);
  auto lay2 = std::make_unique<linear::Layer>(*act1, y_n);

  auto x = std::make_unique<T[]>(lay0->batch_size() * lay0->x_size());
  auto z0 = std::make_unique<T[]>(act0->batch_size() * act0->y_size());
  auto z1 = std::make_unique<T[]>(act1->batch_size() * act1->y_size());
  auto y = std::make_unique<T[]>(lay2->batch_size() * lay2->y_size());

  lay0->xivier(42);
  lay1->xivier(43);
  lay2->xivier(44);

  auto opt0_w = opt::RMSProp(0.001, 0.9);
  auto opt0_b = opt::RMSProp(0.001, 0.9);
  auto opt1_w = opt::RMSProp(0.001, 0.9);
  auto opt1_b = opt::RMSProp(0.001, 0.9);
  auto opt2_w = opt::RMSProp(0.001, 0.9);
  auto opt2_b = opt::RMSProp(0.001, 0.9);

  opt0_w.regist(lay0->weight.get(), lay0->d_weight.get(), lay0->weight_size());
  opt1_w.regist(lay1->weight.get(), lay1->d_weight.get(), lay1->weight_size());
  opt2_w.regist(lay2->weight.get(), lay2->d_weight.get(), lay2->weight_size());
  opt0_b.regist(lay0->bias.get(), lay0->d_bias.get(), lay0->bias_size());
  opt1_b.regist(lay1->bias.get(), lay1->d_bias.get(), lay1->bias_size());
  opt2_b.regist(lay2->bias.get(), lay2->d_bias.get(), lay2->bias_size());

  auto f = [](T x) { return x * x; };
  auto metrics = MSE(batch * y_n);

  int epochs = 100000;
  auto X = std::vector<T>(x_n * batch * epochs);
  auto Y = std::vector<T>(y_n * batch * epochs);
  {
    std::mt19937 engine(42);
    std::normal_distribution<> dist(0.0, 2.0);
    for (int i = 0; i < X.size(); i++) {
      X[i] = dist(engine);
      Y[i] = f(X[i]);
    }
  }

  T score_mean = 1e6;
  for (int e = 0; e < epochs; e++) {
    std::copy_n(&X[e * x_n * batch], x_n * batch, x.get());

    lay0->forward(x.get(), z0.get());
    act0->forward(z0.get(), z0.get());
    lay1->forward(z0.get(), z1.get());
    act1->forward(z1.get(), z1.get());
    lay2->forward(z1.get(), y.get());

    auto score = metrics(y.get(), &Y[e * y_n * batch]);
    printf("%d: score = %.9lf", e, score);
    if ((score_mean = 0.3 * score_mean + (1 - 0.3) * score) < 1e-5) break;
    printf(" score_mean = %lf\n", score_mean);

    metrics.d(y.get(), &Y[e * y_n * batch], y.get());
    lay2->backward(y.get(), z1.get());
    act1->backward(z1.get(), z1.get());
    lay1->backward(z1.get(), z0.get());
    act0->backward(z0.get(), z0.get());
    lay0->backward(z0.get(), x.get());

    opt0_w.update();
    opt0_b.update();
    opt1_w.update();
    opt1_b.update();
    opt2_w.update();
    opt2_b.update();
  }
}

template <typename Opt>
class Model {
 public:
  Model(Opt opt) : base_opt(opt) {}

  void add_layer(std::unique_ptr<LayerBase> lay) {
    if (bufs.size() == 0) {
      bufs.push_back(std::make_unique<T[]>(lay->x_size() * lay->batch_size()));
    }
    if (lay->kind() == Layers::Linear) {
      bufs.push_back(std::make_unique<T[]>(lay->y_size() * lay->batch_size()));
      {
        auto layer = dynamic_cast<linear::Layer*>(lay.get());

        auto opt_w = std::make_unique<Opt>(base_opt);
        auto opt_b = std::make_unique<Opt>(base_opt);

        opt_w->regist(layer->weight.get(), layer->d_weight.get(),
                      layer->weight_size());
        opt_b->regist(layer->bias.get(), layer->d_bias.get(),
                      layer->bias_size());
        opts.push_back(std::move(opt_w));
        opts.push_back(std::move(opt_b));
      }

      x_index.push_back(bufs.size() - 2);
      y_index.push_back(bufs.size() - 1);
    } else if (lay->kind() == Layers::Act) {
      x_index.push_back(bufs.size() - 1);
      y_index.push_back(bufs.size() - 1);
    } else {
      abort();
    }

    layers.push_back(std::move(lay));
  }

  T* x() { return bufs[0].get(); }
  T* y() { return bufs[bufs.size() - 1].get(); }

  LayerBase& top() { return *layers[layers.size() - 1]; }

  void forward() {
    for (int i = 0; i < layers.size(); i++) {
      layers[i]->forward(bufs[x_index[i]].get(), bufs[y_index[i]].get());
    }
  }

  void backward() {
    for (int i = layers.size() - 1; i >= 0; i--) {
      layers[i]->backward(bufs[y_index[i]].get(), bufs[x_index[i]].get());
    }
  }

  void update() {
    for (auto& opt : opts) {
      opt->update();
    }
  }

  std::vector<std::unique_ptr<LayerBase>> layers;
  std::vector<int> x_index;
  std::vector<int> y_index;
  std::vector<std::unique_ptr<T[]>> bufs;
  std::vector<std::unique_ptr<Opt>> opts;

 private:
  Opt base_opt;
};

int main() {
  const int batch = 100, x_n = 1, y_n = 1;

  auto lay0 = std::make_unique<linear::Layer>(batch, x_n, 100);
  auto act0 = std::make_unique<act::Layer<c1func::Tanh>>(*lay0);
  auto lay1 = std::make_unique<linear::Layer>(*act0, 100);
  auto act1 = std::make_unique<act::Layer<c1func::Tanh>>(*lay1);
  auto lay2 = std::make_unique<linear::Layer>(*act1, y_n);

  lay0->xivier(42);
  lay1->xivier(43);
  lay2->xivier(44);

  Model<opt::Adam> model(opt::Adam(0.001, 0.9, 0.999));
  model.add_layer(std::move(lay0));
  model.add_layer(std::move(act0));
  model.add_layer(std::move(lay1));
  model.add_layer(std::move(act1));
  model.add_layer(std::move(lay2));

  auto f = [](T x) { return x * x; };
  auto metrics = MSE(batch * y_n);

  int epochs = 100000;
  auto X = std::vector<T>(x_n * batch * epochs);
  auto Y = std::vector<T>(y_n * batch * epochs);
  {
    std::mt19937 engine(42);
    std::normal_distribution<> dist(0.0, 2.0);
    for (int i = 0; i < X.size(); i++) {
      X[i] = dist(engine);
      Y[i] = f(X[i]);
    }
  }

  T score_mean = 1e6;
  for (int e = 0; e < epochs; e++) {
    std::copy_n(&X[e * x_n * batch], x_n * batch, model.x());
    model.forward();

    auto score = metrics(model.y(), &Y[e * y_n * batch]);
    printf("%d: score = %.9lf", e, score);
    if ((score_mean = 0.3 * score_mean + (1 - 0.3) * score) < 1e-5) break;
    printf(" score_mean = %lf\n", score_mean);

    metrics.d(model.y(), &Y[e * y_n * batch], model.y());
    model.backward();
    model.update();
  }
}
