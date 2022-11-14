#pragma once

#include <memory>
#include <vector>

#include "mynn.h"

using namespace mynn;

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