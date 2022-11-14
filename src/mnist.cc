#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "mynn.h"
#include "mynn_util.h"

using namespace mynn;

class Mnist {
 public:
  std::vector<std::vector<double>> readTrainingFile(std::string filename) {
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    int magic_number = 0;
    int number_of_images = 0;
    int rows = 0;
    int cols = 0;

    if (ifs.fail()) {
      std::cerr << "Failed to open file." << filename << std::endl;
      abort();
    }

    //ヘッダー部より情報を読取る。
    ifs.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    ifs.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    ifs.read((char*)&rows, sizeof(rows));
    rows = reverseInt(rows);
    ifs.read((char*)&cols, sizeof(cols));
    cols = reverseInt(cols);

    std::vector<std::vector<double>> images(number_of_images);
    std::cout << magic_number << " " << number_of_images << " " << rows << " "
              << cols << std::endl;

    for (int i = 0; i < number_of_images; i++) {
      images[i].resize(rows * cols);

      for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
          unsigned char temp = 0;
          ifs.read((char*)&temp, sizeof(temp));
          images[i][rows * row + col] = (double)temp / 256;
        }
      }
    }
    return images;
  }
  std::vector<int> readLabelFile(std::string filename) {
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    int magic_number = 0;
    int number_of_images = 0;

    if (ifs.fail()) {
      std::cerr << "Failed to open file. " << filename << std::endl;
      abort();
    }

    //ヘッダー部より情報を読取る。
    ifs.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    ifs.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);

    std::vector<int> label(number_of_images);

    std::cout << number_of_images << std::endl;

    for (int i = 0; i < number_of_images; i++) {
      unsigned char temp = 0;
      ifs.read((char*)&temp, sizeof(temp));
      label[i] = temp;
    }
    return label;
  }

 private:
  int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
  }
};

class CrossEntropy {
 public:
  CrossEntropy(int b_n, int size) : b_n(b_n), size(size) {}
  T operator()(const T* y_act, const int* y_true) {
    T res = 0;
    for (int b = 0; b < b_n; b++) {
      res -= std::log(y_act[b * size + y_true[b]]);
    }
    return res;
  }
  void d(const T* y_act, const int* y_true, T* dy) {
    for (int b = 0; b < b_n; b++) {
      for (int i = 0; i < size; i++) {
        if (i == y_true[b]) {
          dy[b * size + y_true[b]] =
              -1.0 / (y_act[b * size + y_true[b]] + 1e-12);
        } else {
          dy[b * size + i] = 0.0;
        }
      }
    }
  }

  int size;
  int b_n;
};

template <typename V>
int argmax_n(const V* a, int n) {
  V max = -1000000;
  int res = -1;
  for (int i = 0; i < n; i++) {
    if (max < a[i]) {
      max = a[i];
      res = i;
    }
  }
  return res;
}

int main() {
  const int batch = 100, x_n = 28 * 28, y_n = 10;

  auto lay0 = std::make_unique<linear::Layer>(batch, x_n, 100);
  auto act0 = std::make_unique<act::Layer<c1func::Tanh>>(*lay0);
  auto lay1 = std::make_unique<linear::Layer>(*act0, 100);
  auto act1 = std::make_unique<act::Layer<c1func::Tanh>>(*lay1);
  auto lay2 = std::make_unique<linear::Layer>(*act1, y_n);
  auto act2 = std::make_unique<act::SoftMax>(*lay2);

  lay0->xivier(42);
  lay1->xivier(43);
  lay2->xivier(44);

  Model<opt::Adam> model(opt::Adam(0.001, 0.9, 0.999));
  model.add_layer(std::move(lay0));
  model.add_layer(std::move(act0));
  model.add_layer(std::move(lay1));
  model.add_layer(std::move(act1));
  model.add_layer(std::move(lay2));
  model.add_layer(std::move(act2));

  CrossEntropy metrics(batch, y_n);

  Mnist mnist;
  auto test_X =
      mnist.readTrainingFile("../../data/mnist/t10k-images-idx3-ubyte");
  auto test_Y = mnist.readLabelFile("../../data/mnist/t10k-labels-idx1-ubyte");

  auto evaluate = [&]() {
    T metric_score = 0;
    int correct = 0, N = 0;
    for (int e = 0; e < test_X.size() / batch; e++) {
      int batch_Y[batch];
      for (int b = 0; b < batch; b++) {
        std::copy_n(test_X[e * batch + b].data(), x_n, &(model.x())[b * x_n]);
        batch_Y[b] = test_Y[e * batch + b];
      }

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
    auto test_X =
        mnist.readTrainingFile("../../data/mnist/train-images-idx3-ubyte");
    auto test_Y =
        mnist.readLabelFile("../../data/mnist/train-labels-idx1-ubyte");

    int max_epochs = 1000000;
    std::vector<int> indexs(batch * max_epochs);
    for (int i = 0; i < indexs.size(); i++) {
      indexs[i] = i % test_X.size();
    }
    std::random_shuffle(indexs.begin(), indexs.end());

    T min_score = 1e6;
    int evaluate_count = 1;
    int death_count = 0;
    for (int e = 0; e < max_epochs; e++) {
      int batch_Y[batch];
      for (int b = 0; b < batch; b++) {
        std::copy_n(test_X[indexs[e * batch + b]].data(), x_n,
                    &(model.x())[b * x_n]);
        batch_Y[b] = test_Y[indexs[e * batch + b]];
      }
      model.forward();

      if (evaluate_count == 200) {
        evaluate_count = 1;

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
      } else {
        evaluate_count++;
      }

      metrics.d(model.y(), batch_Y, model.y());
      model.backward();
      model.update();
    }
  }
  evaluate();
}