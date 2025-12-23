#include "idx_dataset.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static uint32_t read_be_u32(std::istream& in) {
  uint8_t b[4]{};
  in.read(reinterpret_cast<char*>(b), 4);
  if (!in) {
    throw std::runtime_error("Unexpected EOF while reading u32");
  }
  return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

static std::vector<float> resize_bilinear_u8_to_f32(
    const std::vector<uint8_t>& src,
    int src_h,
    int src_w,
    int dst_h,
    int dst_w) {
  std::vector<float> dst(size_t(dst_h) * size_t(dst_w), 0.0f);
  if (src_h <= 0 || src_w <= 0 || dst_h <= 0 || dst_w <= 0) {
    return dst;
  }

  const float scale_y = float(src_h) / float(dst_h);
  const float scale_x = float(src_w) / float(dst_w);

  for (int y = 0; y < dst_h; y++) {
    const float src_y = (y + 0.5f) * scale_y - 0.5f;
    int y0 = int(std::floor(src_y));
    int y1 = y0 + 1;
    const float wy1 = src_y - float(y0);
    const float wy0 = 1.0f - wy1;
    if (y0 < 0) { y0 = 0; }
    if (y1 >= src_h) { y1 = src_h - 1; }

    for (int x = 0; x < dst_w; x++) {
      const float src_x = (x + 0.5f) * scale_x - 0.5f;
      int x0 = int(std::floor(src_x));
      int x1 = x0 + 1;
      const float wx1 = src_x - float(x0);
      const float wx0 = 1.0f - wx1;
      if (x0 < 0) { x0 = 0; }
      if (x1 >= src_w) { x1 = src_w - 1; }

      const float v00 = float(src[size_t(y0) * size_t(src_w) + size_t(x0)]);
      const float v01 = float(src[size_t(y0) * size_t(src_w) + size_t(x1)]);
      const float v10 = float(src[size_t(y1) * size_t(src_w) + size_t(x0)]);
      const float v11 = float(src[size_t(y1) * size_t(src_w) + size_t(x1)]);

      const float v0 = wx0 * v00 + wx1 * v01;
      const float v1 = wx0 * v10 + wx1 * v11;
      dst[size_t(y) * size_t(dst_w) + size_t(x)] = wy0 * v0 + wy1 * v1;
    }
  }

  return dst;
}

static void load_idx_images(
    const std::filesystem::path& path,
    int limit,
    int out_height,
    int out_width,
    Matrix& out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Failed to open images file: " + path.string());
  }

  const uint32_t magic = read_be_u32(in);
  if (magic != 2051) {
    throw std::runtime_error("Unexpected images magic: " + std::to_string(magic));
  }
  uint32_t count = read_be_u32(in);
  const uint32_t src_h = read_be_u32(in);
  const uint32_t src_w = read_be_u32(in);

  if (limit > 0 && uint32_t(limit) < count) {
    count = uint32_t(limit);
  }

  out.resize(out_height * out_width, int(count));

  std::vector<uint8_t> buf(size_t(src_h) * size_t(src_w));
  for (uint32_t i = 0; i < count; i++) {
    in.read(reinterpret_cast<char*>(buf.data()), std::streamsize(buf.size()));
    if (!in) {
      throw std::runtime_error("Unexpected EOF while reading image bytes");
    }
    std::vector<float> resized = resize_bilinear_u8_to_f32(buf, int(src_h), int(src_w), out_height, out_width);
    for (int j = 0; j < out_height * out_width; j++) {
      out(j, int(i)) = resized[size_t(j)];
    }
  }
}

static void load_idx_labels(const std::filesystem::path& path, int limit, Matrix& out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Failed to open labels file: " + path.string());
  }

  const uint32_t magic = read_be_u32(in);
  if (magic != 2049) {
    throw std::runtime_error("Unexpected labels magic: " + std::to_string(magic));
  }
  uint32_t count = read_be_u32(in);
  if (limit > 0 && uint32_t(limit) < count) {
    count = uint32_t(limit);
  }

  out.resize(1, int(count));
  for (uint32_t i = 0; i < count; i++) {
    uint8_t lbl = 0;
    in.read(reinterpret_cast<char*>(&lbl), 1);
    if (!in) {
      throw std::runtime_error("Unexpected EOF while reading label bytes");
    }
    out(0, int(i)) = float(lbl);
  }
}

IdxDataset load_mnist_idx_dataset(
    const std::filesystem::path& data_dir,
    int limit,
    int out_height,
    int out_width,
    bool train) {
  const auto images = train ? (data_dir / "train-images-idx3-ubyte") : (data_dir / "t10k-images-idx3-ubyte");
  const auto labels = train ? (data_dir / "train-labels-idx1-ubyte") : (data_dir / "t10k-labels-idx1-ubyte");

  IdxDataset ds;
  ds.height = out_height;
  ds.width = out_width;
  load_idx_images(images, limit, out_height, out_width, ds.images);
  load_idx_labels(labels, limit, ds.labels);

  if (ds.images.cols() != ds.labels.cols()) {
    throw std::runtime_error("Mismatched images/labels counts");
  }
  return ds;
}

