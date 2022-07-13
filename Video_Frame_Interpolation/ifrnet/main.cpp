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
#include "ifrnet_ops.h"

DEFINE_LAYER_CREATOR(Warp)
// 自定义层

int ifrnet(const cv::Mat &in0image, const cv::Mat &in1image, float timestep, cv::Mat &outcv)
// int ifrnet(const ncnn::Mat &in0image, const ncnn::Mat &in1image, float timestep, ncnn::Mat &outimage)
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

    const unsigned char *pixel0data = (const unsigned char *)in0image.data;
    const unsigned char *pixel1data = (const unsigned char *)in1image.data;
    // const int w = in0image.w;
    // const int h = in0image.h;
    const int w = in0image.cols;
    const int h = in0image.rows;
    const int channels = 3; // in0image.elempack;

    //     fprintf(stderr, "%d x %d\n", w, h);

    // pad to 32n
    int w_padded = (w + 31) / 32 * 32;
    int h_padded = (h + 31) / 32 * 32;

    ncnn::Mat in0;
    ncnn::Mat in1;
    in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
    in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
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
                        *outptr++ = *ptr++ * (1 / 255.f) - 0.5f;
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
                        *outptr++ = *ptr++ * (1 / 255.f) - 0.5f;
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
            timestep_padded.create(w_padded / 16, h_padded / 16, 1);
            timestep_padded.fill(timestep);
        }

        // ifrnet
        ncnn::Mat out_padded;
        // 开始做推理
        ncnn::Net net;
        // net.opt.use_vulkan_compute = true;
        // net.opt.use_fp16_packed = true; // 一般与vulakn配套
        // net.opt.use_fp16_storage = true;
        net.register_custom_layer("ifrnet.Warp", Warp_layer_creator); // 特殊处理 开头加上 DEFINE_LAYER_CREATOR(Warp)
        net.load_param("models/ifrnet.param");
        net.load_model("models/ifrnet.bin");
        std::cout << "ifrnet loaded" << std::endl;
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", in0_padded);
        ex.input("in1", in1_padded);
        ex.input("in2", timestep_padded);
        ex.extract("out0", out_padded);
        std::cout << "ifrnet extracted" << std::endl;
        std::cout << "ifrnet out_padded.size = " << out_padded.w << "*" << out_padded.h << "*" << out_padded.c << std::endl;
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
                        *outptr++ = (*ptr++ + 0.5f) * 255.f + 0.5f;
                    }
                }
            }
        }
    }

    // download
    out.to_pixels(outcv.data, ncnn::Mat::PIXEL_BGR2RGB); // 纯RGB现实偏蓝色
    // out.to_pixels((unsigned char *)outimage.data, ncnn::Mat::PIXEL_RGB2BGR);
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
    ifrnet(image0, image1, 0.8, out);
    cv::imshow("image0", image0);
    cv::imshow("image1", image1);
    cv::imshow("out", out);
    cv::waitKey(0);
    return 0;
}

// 已知总帧数，顺序获得每一帧的图片，输出总数已知，获取中间分割时间。===》原始1+原始2===》产生一系列中间图【】，===》结束后写入文件
// 任务处理需要两张图片，给出一个或多个timestep,然后生成新的图片，求该图片顺序

// 已知收尾两张固定，求中间物理位置，根据此设定，在原始fps基础上，扩展对应的物理位置，最终将改集合进行输出
