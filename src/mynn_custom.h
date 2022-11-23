#pragma once

#include "mynn.h"

namespace mynn {
namespace model {
class RNN : public model::ModelBase {
 public:
  RNN(int t, resource::Param& w, resource::Param& b)
      : n(b.size), t(t), w(w), b(b) {
    assert(t >= 1);
    assert(w.size == b.size * b.size);

    for (int i = 0; i < t; i++) {
      affines.emplace_back(std::make_unique<model::Affine>(w, b, true));
      acts.emplace_back(
          std::make_unique<model::Act<c1func::Tanh>>(*affines.back()));
    }
  }

  struct Layer : model::LayerBase {
    Layer(int b_n, RNN& model) : b_n(b_n), model(model) {
      _x = std::make_unique<T[]>(b_n * model.n);

      for (int i = 0; i < model.t; i++) {
        affines.emplace_back(model.affines[i]->create(b_n));
        acts.emplace_back(model.acts[i]->create(b_n));
      }
    }

    void reset() { std::fill_n(affines.front()->input(), b_n * model.n, T(0)); }
    void forward(T* y) {
      for (int i = 0; i < model.t - 1; i++) {
        affines[i]->forward(acts[i]->input());
        if (i == 0) {
          for (int i = 0; i < b_n * model.n; i++) {
            acts[0]->input()[i] += _x[i];
          }
        }
        acts[i]->forward(affines[i + 1]->input());
      }
      affines.back()->forward(acts.back()->input());
      acts.back()->forward(y);
    }
    void backward(const T* dy) {
      std::fill_n(model.w.grad(), model.w.size, T(0));
      std::fill_n(model.b.grad(), model.b.size, T(0));

      acts.back()->backward(dy);
      affines.back()->backward(acts.back()->input());
      for (int i = model.t - 1; i >= 0; i--) {
        acts[i]->backward((i == model.t - 1) ? dy : affines[i + 1]->input());
        affines[i]->backward(acts[i]->input());
        if (i == 0) {
          std::copy_n(acts[i]->input(), b_n * model.n, _x.get());
        }
      }
    }
    T* input() { return _x.get(); }

    int b_n;
    RNN& model;
    std::vector<std::unique_ptr<model::LayerBase>> affines;
    std::vector<std::unique_ptr<model::LayerBase>> acts;
    std::unique_ptr<T[]> _x;
  };

  int y_size() { return n; };
  int x_size() { return n; };
  std::unique_ptr<model::LayerBase> create(int batch) {
    return std::unique_ptr<model::LayerBase>(new Layer(batch, *this));
  }

  int t, n;
  resource::Param& w;
  resource::Param& b;
  std::vector<std::unique_ptr<model::Affine>> affines;
  std::vector<std::unique_ptr<model::Act<c1func::Tanh>>> acts;
};

class Composite : model::ModelBase {
 public:
  Composite() {}
  void add(std::unique_ptr<model::ModelBase>&& model) {
    if (!models.empty()) {
      assert(models.back()->y_size() == model->x_size());
    }
    models.push_back(std::move(model));
  }

  struct Layer : model::LayerBase {
    Layer(int b_n, Composite& model) : b_n(b_n), model(model) {
      for (auto& m : model.models) {
        layers.push_back(m->create(b_n));
      }
    }

    void reset() {
      for (auto& l : layers) {
        l->reset();
      }
    }
    void forward(T* y) {
      assert(layers.size() >= 1);
      for (int i = 0; i < layers.size() - 1; i++) {
        layers[i]->forward(layers[i + 1]->input());
      }
      layers.back()->forward(y);
    }
    void backward(const T* dy) {
      assert(layers.size() >= 1);
      layers.back()->backward(dy);
      for (int i = layers.size() - 2; i >= 0; i--) {
        layers[i]->backward(layers[i + 1]->input());
      }
    }
    T* input() { return layers.front()->input(); }

    int b_n;
    Composite& model;
    std::vector<std::unique_ptr<model::LayerBase>> layers;
  };

  int y_size() { return models.back()->y_size(); };
  int x_size() { return models.front()->x_size(); };
  std::unique_ptr<model::LayerBase> create(int batch) {
    return std::unique_ptr<model::LayerBase>(new Layer(batch, *this));
  }

  std::vector<std::unique_ptr<model::ModelBase>> models;
};
}  // namespace model
}  // namespace mynn