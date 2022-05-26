#include <iostream>
// opencv2
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
// ncnn
#include "net.h"

int main(int argc, char **argv)
{
    cv::Mat image = cv::imread("demo.jpg"); // 读取图片
    if (image.empty())
    {
        std::cout << "read image failed" << std::endl;
        return -1;
    }
    // 推理模型定义
    ncnn::Net net;
    net.load_param("models/pfld-sim.param");
    net.load_model("models/pfld-sim.bin");
    // 前处理
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows, 112, 112);
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = net.create_extractor(); // 创建提取器
    ex.input("input_1", in);                     // 输入
    ncnn::Mat out;
    ex.extract("415", out); // 输出
    // 后处理
    std::vector<float> keypoints;
    keypoints.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        keypoints[j] = out[j] * 112;
    }
    size_t max_len = keypoints.size(); // 画点
    for (size_t i = 0; i < max_len; i += 2)
    {
        cv::circle(image, cv::Point(keypoints[i] * image.cols / 112, keypoints[i + 1] * image.rows / 112), 3, cv::Scalar(255, 255, 255), -1);
    }
    cv::imshow("demo", image);
    cv::waitKey();
    return 0;
}