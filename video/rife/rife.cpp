// opencv2
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// c++
#include <vector>
#include <iostream>
#include <format>
// ncnn
#include "net.h"
// #include "layer.h" // 用于处理自定义的层
// #include "pipeline.h"
#include "rife_ops.h"

DEFINE_LAYER_CREATOR(Warp)
// 自定义层
// class Warp : public ncnn::Layer
// {
// public:
//     Warp();
//     virtual int create_pipeline(const ncnn::Option &opt);
//     virtual int destroy_pipeline(const ncnn::Option &opt);
//     virtual int forward(const std::vector<ncnn::Mat> &bottom_blobs, std::vector<ncnn::Mat> &top_blobs, const ncnn::Option &opt) const;
//     virtual int forward(const std::vector<ncnn::VkMat> &bottom_blobs, std::vector<ncnn::VkMat> &top_blobs, ncnn::VkCompute &cmd, const ncnn::Option &opt) const;

// private:
//     ncnn::Pipeline *pipeline_warp;
//     ncnn::Pipeline *pipeline_warp_pack4;
//     ncnn::Pipeline *pipeline_warp_pack8;
// };

int rife4(const cv::Mat &in0image, const cv::Mat &in1image, float timestep, cv::Mat &outcv)
{
    if (timestep == 0.f)
    {
        // outimage = in0image;
        return 0;
    }

    if (timestep == 1.f)
    {
        // outimage = in1image;
        return 0;
    }
    // 推理网络定义
    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    net.register_custom_layer("rife.Warp", Warp_layer_creator); // 特殊处理 开头加上 DEFINE_LAYER_CREATOR(Warp)
    net.load_param("models/flownet.param");
    net.load_model("models/flownet.bin");
    // 特殊图层处理
    // 图像转换
    const unsigned char *pixel0data = (const unsigned char *)in0image.data;
    const unsigned char *pixel1data = (const unsigned char *)in1image.data;
    const int w = in0image.cols;
    const int h = in0image.rows;
    const int channels = 3; // in0image.elempack;
    // pad to 32n
    int w_padded = (w + 31) / 32 * 32;
    int h_padded = (h + 31) / 32 * 32;
    // 入口处理
    ncnn::Mat in0;
    ncnn::Mat in1;
    in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
    in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
    // 输出
    ncnn::Mat out;
    {
        // preproc and border padding
        ncnn::Mat in0_padded;
        ncnn::Mat in1_padded;
        ncnn::Mat timestep_padded;
        {
            in0_padded.create(w_padded, h_padded, 3);
            for (int q = 0; q < 3; q++)
            {
                float *outptr = in0_padded.channel(q);

                int i = 0;
                for (; i < h; i++)
                {
                    const float *ptr = in0.channel(q).row(i);

                    int j = 0;
                    for (; j < w; j++)
                    {
                        *outptr++ = *ptr++ * (1 / 255.f);
                    }
                    for (; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
                for (; i < h_padded; i++)
                {
                    for (int j = 0; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
            }
        }
        {
            in1_padded.create(w_padded, h_padded, 3);
            for (int q = 0; q < 3; q++)
            {
                float *outptr = in1_padded.channel(q);

                int i = 0;
                for (; i < h; i++)
                {
                    const float *ptr = in1.channel(q).row(i);

                    int j = 0;
                    for (; j < w; j++)
                    {
                        *outptr++ = *ptr++ * (1 / 255.f);
                    }
                    for (; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
                for (; i < h_padded; i++)
                {
                    for (int j = 0; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
            }
        }
        {
            timestep_padded.create(w_padded, h_padded, 1);
            timestep_padded.fill(timestep);
        }
        // flownet
        ncnn::Mat out_padded;
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", in0_padded);
        ex.input("in1", in1_padded);
        ex.input("in2", timestep_padded);
        ex.extract("out0", out_padded);
        // std::cout << "out shape" << out_padded.w << " " << out_padded.h << std::endl;
        // cut padding and postproc
        out.create(w, h, 3);
        {
            for (int q = 0; q < 3; q++)
            {
                float *outptr = out.channel(q);
                const float *ptr = out_padded.channel(q);

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        *outptr++ = *ptr++ * 255.f + 0.5f;
                    }
                }
            }
        }
    }
    // outcv(in.h, in.w, CV_8UC3);
    out.to_pixels(outcv.data, ncnn::Mat::PIXEL_BGR2RGB); // 纯RGB现实偏蓝色
    return 0;
}

int main(int argc, char **argv)
{
    cv::Mat image0 = cv::imread("0.png"); // 输入一张图片，BGR格式
    cv::Mat image1 = cv::imread("1.png"); // 输入一张图片，BGR格式
    if (image0.empty() || image1.empty())
    {
        std::cout << "read image failed" << std::endl;
        return -1;
    }
    cv::Mat out(image0.rows, image0.cols, CV_8UC3);
    rife4(image0, image1, 0.5, out);
    cv::imshow("out", out);
    cv::waitKey(0);
    return 0;
}
