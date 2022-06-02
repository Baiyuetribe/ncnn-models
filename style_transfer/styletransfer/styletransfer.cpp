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
    net.opt.use_vulkan_compute = true; // 开启GPU加速
    net.load_param("models/candy9.param");
    net.load_model("models/candy9.bin");
    // 前处理
    const int w = image.cols;
    const int h = image.rows;
    const int target_size = 1000; // 值越大细节表现越好，但是速度越慢
    int target_w = w;
    int target_h = h;
    if (w < h)
    {
        target_h = target_size;
        target_w = target_size * w / h;
    }
    else
    {
        target_w = target_size;
        target_h = target_size * h / w;
    }
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, w, h, target_w, target_h);
    ncnn::Mat out;
    ncnn::Extractor ex = net.create_extractor();
    ex.input("input1", in);
    ex.extract("output1", out);
    // 后处理
    cv::Mat outbgr;
    outbgr.create(out.h, out.w, CV_8UC3);
    out.to_pixels(outbgr.data, ncnn::Mat::PIXEL_RGB2BGR);

    // cv::imwrite("out.png", outbgr);    // 保存图片
    cv::imshow("out", outbgr);
    cv::waitKey(0);
    return 0;
}