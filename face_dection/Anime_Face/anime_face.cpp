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
    net.load_param("models/anime-face_hrnetv2.param");
    net.load_model("models/anime-face_hrnetv2.bin");
    // 前处理
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows, 256, 256);
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
    // const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals); // 可能有误

    ncnn::Extractor ex = net.create_extractor(); // 创建提取器
    ex.input("img", in);                         // 输入
    ncnn::Mat out;
    ex.extract("heatmap", out);                                         // 输出
    std::cout << out.w << " " << out.h << " " << out.dims << std::endl; // 28 * 64 * 64
    // 后处理 -- 欢迎修正

    // std::vector<float> keypoints;
    // keypoints.resize(out.w);
    // for (int j = 0; j < out.w; j++)
    // {
    //     // std::cout << out[j] << std::endl;
    //     keypoints[j] = out[j] * 256; // 一次存储两个节点位置
    // }
    // size_t max_len = keypoints.size(); // 画点
    // cv::Mat res;
    // cv::resize(image, res, cv::Size(256, 256));
    // // for (size_t i = 0; i < max_len; i += 2)
    // for (size_t i = 0; i < max_len; i += 2)
    // {
    //     cv::circle(res, cv::Point(keypoints[i] * image.cols / 256, keypoints[i + 1] * image.rows / 256), 3, cv::Scalar(255, 255, 255), -1);
    // }
    // cv::imshow("demo", res);
    // cv::waitKey();
    return 0;
}