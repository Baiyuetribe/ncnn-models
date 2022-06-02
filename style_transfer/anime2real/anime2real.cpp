// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// ncnn
#include "gpu.h"
#include "net.h"
#include <iostream>

int main(int argc, char **argv)
{
    cv::Mat image = cv::imread("input.png");
    if (image.empty())
    {
        std::cout << "cv::imread failed" << std::endl;
        return -1;
    }
    ncnn::Net net;
    net.opt.use_vulkan_compute = true;       // 开启GPU加速
    net.load_param("models/netG_A2B.param"); // netG_A2B OR netG_B2A
    net.load_model("models/netG_A2B.bin");
    // 前处理
    const int w = image.cols;
    const int h = image.rows;
    int target_w = w;
    int target_h = h;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, w, h, 512, 512);
    // const float mean_vals[3] = {0.5f * 255.f, 0.5f * 255.f, 0.5f * 255.f}; // transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    // const float norm_vals[3] = {1 / 0.5f / 255.f, 1 / 0.5f / 255.f, 1 / 0.5f / 255.f};
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Mat out;
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);
    std::cout << "input shape:" << in.w << " " << in.h << " " << in.c << std::endl;
    ex.extract("out0", out);
    std::cout << "output shape:" << out.w << " " << out.h << " " << out.c << std::endl;
    // 后处理
    cv::Mat result(out.h, out.w, CV_32FC3);
    for (int i = 0; i < out.c; i++)
    {
        float *out_data = out.channel(i);
        for (int h = 0; h < out.h; h++)
        {
            for (int w = 0; w < out.w; w++)
            {
                result.at<cv::Vec3f>(h, w)[2 - i] = out_data[h * out.h + w];
            }
        }
    }
    cv::Mat result8U(out.h, out.w, CV_8UC3);
    result.convertTo(result8U, CV_8UC3, 127.5, 127.5);

    // cv::imwrite("out.png", outbgr);    // 保存图片
    cv::imshow("in", image);
    cv::imshow("out", result8U);
    cv::waitKey(0);
    return 0;
}