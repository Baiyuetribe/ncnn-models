#include "net.h" // ncnn
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "iostream"

int main(int argc, char **argv)
{
    cv::Mat m = cv::imread("input.png"); // 输入一张图片，BGR格式
    if (m.empty())
    {
        std::cout << "read image failed" << std::endl;
        return -1;
    }

    ncnn::Net net;
    net.opt.use_vulkan_compute = true;                 // GPU环境
    net.load_param("models/deeplabv3_resnet50.param"); // 模型加载 or resnet101 or deeplabv3_mobilenet_v3_large
    net.load_model("models/deeplabv3_resnet50.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(m.data, ncnn::Mat::PIXEL_BGR2RGB, m.cols, m.rows, 224, 224); // 图片缩放
    const float mean_vals[3] = {0.485f / 255.f, 0.456f / 255.f, 0.406f / 255.f};                              // Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    const float norm_vals[3] = {1.0 / 0.229f / 255.f, 1.0 / 0.224f / 255.f, 1.0 / 0.225f / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals); // 像素范围0~255

    ncnn::Mat out;
    ncnn::Extractor ex = net.create_extractor();
    ex.input("input.1", in); // 输入
    ex.extract("607", out);  // 输出
    // print shape(out);
    std::cout << out.w << out.h << out.dims << out.c << std::endl; // 输出维度 24*224*224 shape (21, H, W)
    // 后处理待写
    return 0;
}