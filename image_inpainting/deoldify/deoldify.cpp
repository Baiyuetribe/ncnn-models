#include "net.h" // ncnn
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <algorithm> // min/max
#include "iostream"

float get_max(ncnn::Mat &out)
{
    int out_c = out.c;
    int out_w = out.w;
    int out_h = out.h;
    for (int i = 0; i < out_c; i++)
    {
        float max = -100, min = 9999999999999;
        for (int j = 0; j < out_h; j++)
        {
            for (int k = 0; k < out_w; k++)
            {
                if (out.channel(i).row(j)[k] > max)
                {
                    max = out.channel(i).row(j)[k];
                }
                if (out.channel(i).row(j)[k] < min)
                {
                    min = out.channel(i).row(j)[k];
                }
            }
        }
        // std::cout << max << " " << min << std::endl;
    }
    return 0;
}

int main(int argc, char **argv)
{
    cv::Mat m = cv::imread("input.png"); // 输入一张图片，BGR格式
    if (m.empty())
    {
        std::cout << "read image failed" << std::endl;
        return -1;
    }

    ncnn::Net net;
    net.opt.use_vulkan_compute = true;           // GPU环境
    net.load_param("models/deoldify.256.param"); // 模型加载
    net.load_model("models/deoldify.256.bin");

    //    ncnn::Mat in = ncnn::Mat::from_pixels_resize(m.data, ncnn::Mat::PIXEL_RGB, m.cols, m.rows, 512, 512);
    //    ncnn::Mat in = ncnn::Mat::from_pixels_resize(m.data, ncnn::Mat::PIXEL_RGB, m.cols, m.rows, 128, 128);
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(m.data, ncnn::Mat::PIXEL_BGR2RGB, m.cols, m.rows, 256, 256); // 图片缩放
    //    const float pre_mean_vals[3] = {0.f, 0.f, 0.f};
    //    const float pre_norm_vals[3] = {1.0/255.0f,1.0/255.0f,1.0/255.0f};
    //    in.substract_mean_normalize(pre_mean_vals, pre_norm_vals);
    //    const float mean_vals[3] = {0.485f, 0.456f, 0.406f};
    //    const float norm_vals[3] = {0.229f, 0.224f, 0.225f};
    //    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Mat out;
    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", in);  // 输入
    ex.extract("out", out); // 输出
    get_max(out);

    //    cv::Mat cv_out = cv::Mat::zeros(512, 512, CV_8UC3);
    //    cv::Mat cv_out = cv::Mat::zeros(128, 128, CV_8UC3);
    cv::Mat cv_out = cv::Mat::zeros(256, 256, CV_8UC3);

    //    out.to_pixels(cv_out.data, ncnn::Mat::PIXEL_RGB);
    //    out.to_pixels(cv_out.data, ncnn::Mat::PIXEL_BGR);
    for (int c = 0; c < 3; c++)
    {
        for (int i = 0; i < out.h; i++)
        {
            for (int j = 0; j < out.w; j++)
            {
                float t = ((float *)out.data)[j + i * out.w + c * out.h * out.w];
                cv_out.data[(2 - c) + j * 3 + i * out.w * 3] = t;
            }
        }
    }

    cv::imwrite("ncnn.jpg", cv_out);
    cv::imshow("output", cv_out);
    cv::waitKey(0);

    return 0;
}