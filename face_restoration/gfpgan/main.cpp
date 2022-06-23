#include <opencv2/opencv.hpp>

#include <iostream>
#include "gfpgan.h" // 自定义

static void to_ocv(const ncnn::Mat &result, cv::Mat &out)
{
    cv::Mat cv_result_32F = cv::Mat::zeros(cv::Size(512, 512), CV_32FC3);
    for (int i = 0; i < result.h; i++)
    {
        for (int j = 0; j < result.w; j++)
        {
            cv_result_32F.at<cv::Vec3f>(i, j)[2] = (result.channel(0)[i * result.w + j] + 1) / 2;
            cv_result_32F.at<cv::Vec3f>(i, j)[1] = (result.channel(1)[i * result.w + j] + 1) / 2;
            cv_result_32F.at<cv::Vec3f>(i, j)[0] = (result.channel(2)[i * result.w + j] + 1) / 2;
        }
    }

    cv::Mat cv_result_8U;
    cv_result_32F.convertTo(cv_result_8U, CV_8UC3, 255.0, 0);
    cv_result_8U.copyTo(out);
}

// 先测试gfpan
int main(int argc, char **argv)
{
    std::string in_path = "in.png";
    std::string out_path = "out.png";
    cv::Mat img = cv::imread(in_path);
    if (img.empty())
    {
        std::cout << "image not found" << std::endl;
        return -1;
    }
    // std::cout << "入口0 size: " << img.size() << std::endl;
    // 测试gfpgan
    GFPGAN gfpgan;
    // gfpgan.load("models/encoder.param", "models/encoder.bin", "models/style.bin"); // 单张处理4s，但是GPU消耗很小
    static std::string exe_path = getExeDir();
    gfpgan.load("models/encoder.param", "models/encoder.bin", "models/style.bin"); // 单张处理4s，但是GPU消耗很小
    ncnn::Mat res3;
    auto start = std::chrono::system_clock::now();
    gfpgan.process(img, res3); // 入口任意，出口都是512*512
    auto end = std::chrono::system_clock::now();
    // 汇总时间
    std::chrono::duration<double> elapsed_seconds = (end - start);
    std::cout << "Total elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
    // std::cout << "res3 size" << res3.w << "h:" << res3.h << std::endl;
    // 转换为opencv格式
    cv::Mat res3_cv;
    to_ocv(res3, res3_cv);
    // cv::imshow("res3", res3_cv);
    // cv::waitKey(0);
    cv::imwrite(out_path, res3_cv); // 保存结果
    return 0;
}