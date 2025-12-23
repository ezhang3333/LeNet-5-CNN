#pragma once

#include <cstdint>
#include <filesystem>

#include "utils.h"

struct IdxDataset {
  Matrix images;  // (H*W, N), uint8 converted to float
  Matrix labels;  // (1, N), label in [0,9] as float
  int height = 0;
  int width = 0;
};

IdxDataset load_mnist_idx_dataset(
    const std::filesystem::path& data_dir,
    int limit,
    int out_height,
    int out_width,
    bool train);

