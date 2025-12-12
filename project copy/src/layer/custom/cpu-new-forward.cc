#include "cpu-new-forward.h"

void conv_forward_cpu(
    float *output,
    const float *input,
    const float *mask,
    const int Batch,
    const int Map_out,
    const int Channel,
    const int Height,
    const int Width,
    const int K)
{
    // Output spatial size (no padding, stride=1)
    const int Height_out = Height - K + 1;
    const int Width_out  = Width  - K + 1;

    // Indexing helpers (NCHW for input; N M H W for output)
    #define out_4d(b, m, h, w) output[(b) * (Map_out * Height_out * Width_out) \
                                   + (m) * (Height_out * Width_out) \
                                   + (h) * (Width_out) + (w)]
    #define in_4d(b, c, h, w)  input[(b) * (Channel * Height * Width) \
                                   + (c) * (Height * Width) \
                                   + (h) * (Width) + (w)]
    #define mask_4d(m, c, p, q) mask[(m) * (Channel * K * K) \
                                   + (c) * (K * K) \
                                   + (p) * (K) + (q)]

    // Naive, correct CPU reference (seven nested loops)
    for (int b = 0; b < Batch; ++b) {
        for (int m = 0; m < Map_out; ++m) {
            for (int h = 0; h < Height_out; ++h) {
                for (int w = 0; w < Width_out; ++w) {
                    float acc = 0.0f;
                    for (int c = 0; c < Channel; ++c) {
                        for (int p = 0; p < K; ++p) {
                            for (int q = 0; q < K; ++q) {
                                acc += in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
                            }
                        }
                    }
                    out_4d(b, m, h, w) = acc;
                }
            }
        }
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}