// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// ncnn
#include "gpu.h"
#include "net.h"
#include <iostream>

static int styletransfer(const ncnn::Net &net, const cv::Mat &bgr, cv::Mat &outbgr)
{
    const int w = bgr.cols;
    const int h = bgr.rows;

    int target_w = 512;
    int target_h = 512;

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, w, h, target_w, target_h);
    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Mat out;
    {
        ncnn::Extractor ex = net.create_extractor();
        ex.input("input", in);
        ex.extract("out", out);
    }

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
    result8U.copyTo(outbgr);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 3) // 输入和输出
    {
        std::cerr << "Usage: " << argv[0] << " <image_path> <out_path>" << std::endl;
        return -1;
    }

    std::string imagepath = argv[1];
    std::string outpath = argv[2];

    cv::Mat bgr = cv::imread(imagepath, 1);
    if (bgr.empty())
    {
        std::cerr << "cv::imread failed" << std::endl;
        return -1;
    }

    ncnn::Net net;                                         // 定义神经网络
    stylenet.opt.use_vulkan_compute = true;                // 开启GPU加速
    stylenet.load_param("models/face_paint_512_v2.param"); // 加载模型参数
    stylenet.load_model("models/face_paint_512_v2.param");
    // 开始推理
    cv::Mat outbgr;
    styletransfer(stylenet, bgr, outbgr); // 具体推理
    // 展示输出
    cv::imwrite(outpath, outbgr); // 保存结果
    cv::imshow("out", outbgr);    // 保存结果
    cv::waitKey(0);
    return 0;
}
