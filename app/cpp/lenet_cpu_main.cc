#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "idx_dataset.h"

#include "layer/conv.h"
#include "layer/fully_connected.h"
#include "layer/max_pooling.h"
#include "layer/relu.h"
#include "layer/softmax.h"
#include "loss/cross_entropy_loss.h"
#include "network.h"
#include "optimizer/sgd.h"
#include "utils.h"

static std::filesystem::path find_repo_root(std::filesystem::path start) {
  std::error_code ec;
  start = std::filesystem::absolute(start, ec);
  if (ec) {
    start.clear();
  }

  auto looks_like_root = [&](const std::filesystem::path& p) -> bool {
    return std::filesystem::exists(p / "CMakeLists.txt") &&
           std::filesystem::exists(p / "app") &&
           std::filesystem::exists(p / "cnn");
  };

  std::filesystem::path cur = start.empty() ? std::filesystem::current_path() : start;
  for (int i = 0; i < 10; i++) {
    if (looks_like_root(cur)) {
      return cur;
    }
    if (!cur.has_parent_path()) {
      break;
    }
    cur = cur.parent_path();
  }
  return {};
}

static Network create_lenet86() {
  Network net;
  net.add_layer(new Conv(1, 86, 86, 4, 7, 7, 1, 0, 0));
  net.add_layer(new ReLU());
  net.add_layer(new MaxPooling(4, 80, 80, 2, 2, 2));

  net.add_layer(new Conv(4, 40, 40, 16, 7, 7, 1, 0, 0));
  net.add_layer(new ReLU());
  net.add_layer(new MaxPooling(16, 34, 34, 4, 4, 4));

  net.add_layer(new FullyConnected(16 * 9 * 9, 32));
  net.add_layer(new ReLU());
  net.add_layer(new FullyConnected(32, 10));
  net.add_layer(new Softmax());

  net.add_loss(new CrossEntropy());
  return net;
}

static void usage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " [--data <mnist_dir>] [--out <weights.bin>] [options]\n"
      << "\nOptions:\n"
      << "  --data <mnist_dir>  directory containing MNIST IDX files\n"
      << "  --out <weights.bin> output path (default: <repo_root>/weights.bin)\n"
      << "  --epochs N          (default: 1)\n"
      << "  --batch N           (default: 64)\n"
      << "  --lr LR             (default: 0.01)\n"
      << "  --train-limit N     (default: 60000)\n"
      << "  --test-limit N      (default: 10000)\n"
      << "  --resume <weights>  load weights before training\n";
}

static std::filesystem::path default_data_dir() {
  const std::filesystem::path repo_root = find_repo_root(std::filesystem::current_path());

  if (const char* env = std::getenv("MNIST_DIR")) {
    if (env[0] != '\0') {
      return std::filesystem::path(env);
    }
  }
  if (const char* env = std::getenv("LENET_DATA")) {
    if (env[0] != '\0') {
      return std::filesystem::path(env);
    }
  }

  const std::filesystem::path candidates[] = {
      std::filesystem::path("cnn") / "datasets",
      std::filesystem::path("data") / "mnist",
      std::filesystem::path("mnist"),
      repo_root / "cnn" / "datasets",
      repo_root / "data" / "mnist",
      repo_root / "mnist",
  };
  for (const auto& p : candidates) {
    if (!p.empty() && std::filesystem::exists(p)) {
      return p;
    }
  }
  return {};
}

static const char* require_arg(int& i, int argc, char** argv) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("Missing value for ") + argv[i]);
  }
  i++;
  return argv[i];
}

static Matrix gather_cols(const Matrix& src, const std::vector<int>& indices, int start, int count) {
  if (count <= 0) {
    return Matrix(src.rows(), 0);
  }
  Matrix out(src.rows(), count);
  for (int i = 0; i < count; i++) {
    out.col(i) = src.col(indices[size_t(start + i)]);
  }
  return out;
}

static int count_correct(const Matrix& predictions, const Matrix& labels) {
  const int n = predictions.cols();
  int correct = 0;
  for (int i = 0; i < n; i++) {
    Matrix::Index max_index;
    predictions.col(i).maxCoeff(&max_index);
    correct += int(max_index) == int(labels(0, i));
  }
  return correct;
}

int main(int argc, char** argv) {
  try {
    const std::filesystem::path cwd = std::filesystem::current_path();
    const std::filesystem::path repo_root = find_repo_root(cwd);

    std::filesystem::path data_dir = default_data_dir();
    std::filesystem::path out_weights;
    std::filesystem::path resume_weights;

    int epochs = 1;
    int batch = 64;
    float lr = 0.01f;
    int train_limit = 60000;
    int test_limit = 10000;

    for (int i = 1; i < argc; i++) {
      const std::string arg = argv[i];
      if (arg == "--help" || arg == "-h") {
        usage(argv[0]);
        return 0;
      }
      if (arg == "--data") {
        data_dir = require_arg(i, argc, argv);
        continue;
      }
      if (arg == "--out") {
        out_weights = require_arg(i, argc, argv);
        continue;
      }
      if (arg == "--resume") {
        resume_weights = require_arg(i, argc, argv);
        continue;
      }
      if (arg == "--epochs") {
        epochs = std::stoi(require_arg(i, argc, argv));
        continue;
      }
      if (arg == "--batch") {
        batch = std::stoi(require_arg(i, argc, argv));
        continue;
      }
      if (arg == "--lr") {
        lr = std::stof(require_arg(i, argc, argv));
        continue;
      }
      if (arg == "--train-limit") {
        train_limit = std::stoi(require_arg(i, argc, argv));
        continue;
      }
      if (arg == "--test-limit") {
        test_limit = std::stoi(require_arg(i, argc, argv));
        continue;
      }
      throw std::runtime_error("Unknown arg: " + arg);
    }

    if (out_weights.empty()) {
      if (!repo_root.empty()) {
        out_weights = repo_root / "weights.bin";
      } else {
        out_weights = cwd / "weights.bin";
      }
      std::cout << "No --out provided; defaulting to: " << out_weights.string() << std::endl;
    }

    if (data_dir.empty()) {
      usage(argv[0]);
      std::cerr << "\nHint: pass --data <mnist_dir> (folder with MNIST IDX files), or set MNIST_DIR / LENET_DATA.\n";
      std::cerr << "Current working directory: " << cwd.string() << std::endl;
      if (!repo_root.empty()) {
        std::cerr << "Detected repo root: " << repo_root.string() << std::endl;
      }
      return 2;
    }

    std::cout << "Loading MNIST IDX files from: " << data_dir.string() << std::endl;
    IdxDataset train = load_mnist_idx_dataset(data_dir, train_limit, 86, 86, true);
    IdxDataset test = load_mnist_idx_dataset(data_dir, test_limit, 86, 86, false);

    Network net = create_lenet86();
    if (!resume_weights.empty()) {
      std::cout << "Loading weights: " << resume_weights.string() << std::endl;
      net.load_parameters(resume_weights.string());
    }

    SGD opt(lr);

    std::cout << "Train samples: " << train.images.cols() << ", test samples: " << test.images.cols() << std::endl;
    std::cout << "Training: epochs=" << epochs << " batch=" << batch << " lr=" << lr << std::endl;

    const int n_train = train.images.cols();
    std::vector<int> indices;
    indices.resize(static_cast<size_t>(n_train));
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(0xC0FFEE);

    for (int epoch = 1; epoch <= epochs; epoch++) {
      std::shuffle(indices.begin(), indices.end(), rng);

      auto t0 = std::chrono::high_resolution_clock::now();
      for (int start = 0; start < n_train; start += batch) {
        const int b = std::min(batch, n_train - start);
        const Matrix x = gather_cols(train.images, indices, start, b);
        const Matrix y_labels = gather_cols(train.labels, indices, start, b);
        const Matrix y = one_hot_encode(y_labels, 10);

        net.forward(x);
        net.backward(x, y);
        net.update(opt);
      }

      const auto t1 = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double> dt = t1 - t0;

      int correct = 0;
      const int n_test = test.images.cols();
      const int eval_batch = std::max(1, std::min(512, batch * 4));
      for (int start = 0; start < n_test; start += eval_batch) {
        const int b = std::min(eval_batch, n_test - start);
        const Matrix x = test.images.middleCols(start, b);
        const Matrix y = test.labels.middleCols(start, b);
        net.forward(x);
        correct += count_correct(net.output(), y);
      }
      const float acc = n_test > 0 ? float(correct) / float(n_test) : 0.0f;
      std::cout << "Epoch " << epoch << "/" << epochs << " - test accuracy: " << acc
                << " - time: " << dt.count() << "s" << std::endl;
    }

    std::cout << "Saving weights to: " << out_weights.string() << std::endl;
    net.save_parameters(out_weights.string());
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
