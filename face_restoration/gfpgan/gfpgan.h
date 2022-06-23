// gfpgan implemented with ncnn library

#ifndef GFPGAN_H
#define GFPGAN_H

#include <vector>
#include <ostream>
#include <random>
#include <chrono>
#include <stdio.h>
#include <fstream>
#include <opencv2/opencv.hpp>

// ncnn
#include "net.h"
#include "cpu.h"
#include "layer.h"

typedef struct StyleConvWeights
{
    int data_size;
    int inc;
    int hid_dim;
    int num_output;
    std::vector<float> style_convs_modulated_conv_weight;
    std::vector<float> style_convs_modulated_conv_modulation_weight;
    std::vector<float> style_convs_modulated_conv_modulation_bias;
    std::vector<float> style_convs_weight;
    std::vector<float> style_convs_bias;
} StyleConvWeights;
typedef struct ToRgbConvWeights
{
    int data_size;
    int inc;
    int hid_dim;
    int num_output;
    std::vector<float> to_rgbs_modulated_conv_weight;
    std::vector<float> to_rgbs_modulated_conv_modulation_weight;
    std::vector<float> to_rgbs_modulated_conv_modulation_bias;
    std::vector<float> to_rgbs_bias;
} ToRgbConvWeights;
const int style_conv_sizes[][5] = {{512 * 512 * 3 * 3, 512 * 512, 512, 1, 512},  // 0
                                   {512 * 512 * 3 * 3, 512 * 512, 512, 1, 512},  // 1
                                   {512 * 512 * 3 * 3, 512 * 512, 512, 1, 512},  // 2
                                   {512 * 512 * 3 * 3, 512 * 512, 512, 1, 512},  // 3
                                   {512 * 512 * 3 * 3, 512 * 512, 512, 1, 512},  // 4
                                   {512 * 512 * 3 * 3, 512 * 512, 512, 1, 512},  // 5
                                   {512 * 512 * 3 * 3, 512 * 512, 512, 1, 512},  // 6
                                   {512 * 512 * 3 * 3, 512 * 512, 512, 1, 512},  // 7
                                   {256 * 512 * 3 * 3, 512 * 512, 512, 1, 256},  // 8
                                   {256 * 256 * 3 * 3, 256 * 512, 256, 1, 256},  // 9
                                   {128 * 256 * 3 * 3, 256 * 512, 256, 1, 128},  // 10
                                   {128 * 128 * 3 * 3, 128 * 512, 128, 1, 128},  // 11
                                   {64 * 128 * 3 * 3, 128 * 512, 128, 1, 64},    // 12
                                   {64 * 64 * 3 * 3, 64 * 512, 64, 1, 64},       // 13
                                   {512 * 512 * 3 * 3, 512 * 512, 512, 1, 512}}; // 14
const int to_rgb_sizes[][4] = {{3 * 512 * 1 * 1, 512 * 512, 512, 3},             // 0
                               {3 * 512 * 1 * 1, 512 * 512, 512, 3},             // 1
                               {3 * 512 * 1 * 1, 512 * 512, 512, 3},             // 2
                               {3 * 512 * 1 * 1, 512 * 512, 512, 3},             // 3
                               {3 * 256 * 1 * 1, 256 * 512, 256, 3},             // 4
                               {3 * 128 * 1 * 1, 128 * 512, 128, 3},             // 5
                               {3 * 64 * 1 * 1, 64 * 512, 64, 3},                // 6
                               {3 * 512 * 1 * 1, 512 * 512, 512, 3}};            // 7
const int style_conv_channels[][3] = {{512, 512, 512},                           // 0
                                      {512, 512, 512},                           // 1
                                      {512, 512, 512},                           // 2
                                      {512, 512, 512},                           // 3
                                      {512, 512, 512},                           // 4
                                      {512, 512, 512},                           // 5
                                      {512, 512, 512},                           // 6
                                      {512, 512, 512},                           // 7
                                      {512, 512, 256},                           // 8
                                      {512, 256, 256},                           // 9
                                      {512, 256, 128},                           // 10
                                      {512, 128, 128},                           // 11
                                      {512, 128, 64},                            // 12
                                      {512, 64, 64},                             // 13
                                      {512, 512, 512}};                          // 14

const int to_rgb_channels[][3] = {{512, 512, 3},  // 0
                                  {512, 512, 3},  // 1
                                  {512, 512, 3},  // 2
                                  {512, 512, 3},  // 3
                                  {512, 256, 3},  // 4
                                  {512, 128, 3},  // 5
                                  {512, 64, 3},   // 6
                                  {512, 512, 3}}; // 7

class GFPGAN
{
public:
    GFPGAN();
    ~GFPGAN();

    int load(const std::string &param_path, const std::string &model_path, const std::string &style_path);

    int process(const cv::Mat &img, ncnn::Mat &outimage);

private:
    int modulated_conv(ncnn::Mat &x, ncnn::Mat &style,
                       const float *self_weight, const float *weights,
                       const float *bias, int sample_mode, int demodulate,
                       int inc, int num_output, int kernel_size, int hid_dim, ncnn::Mat &out);
    int to_rgbs(ncnn::Mat &out, ncnn::Mat &latent, ncnn::Mat &skip,
                int inc, int hid_dim, int num_output,
                const float *to_rgbs_modulated_conv_weight,
                const float *to_rgbs_modulated_conv_modulation_weight,
                const float *to_rgbs_modulated_conv_modulation_bias,
                const float *to_rgbs_bias);
    int style_convs_modulated_conv(ncnn::Mat &x, ncnn::Mat style, int sample_mode,
                                   int demodulate, ncnn::Mat &out, int inc, int hid_dim, int num_output,
                                   const float *style_convs_modulated_conv_weight,
                                   const float *style_convs_modulated_conv_modulation_weight,
                                   const float *style_convs_modulated_conv_modulation_bias,
                                   const float *style_convs_weight,
                                   const float *style_convs_bias);
    int load_weights(const char *model_path, std::vector<StyleConvWeights> &style_conv_weights,
                     std::vector<ToRgbConvWeights> &to_rgbs_conv_weights, ncnn::Mat &const_input);

private:
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
    std::vector<StyleConvWeights> style_conv_weights;
    std::vector<ToRgbConvWeights> to_rgbs_conv_weights;
    ncnn::Mat const_input;
    ncnn::Net net;
};

#endif // GFPGAN_H
