#include "net.h" // ncnn
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "iostream"

static int print_topk(const std::vector<float> &cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int>> vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int>>());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
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
    net.load_param("models/mobilenet_v3.param"); // 模型加载
    net.load_model("models/mobilenet_v3.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(m.data, ncnn::Mat::PIXEL_BGR2RGB, m.cols, m.rows, 224, 224); // 图片缩放
    const float mean_vals[3] = {0.485f / 255.f, 0.456f / 255.f, 0.406f / 255.f};                              // Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    const float norm_vals[3] = {1.0 / 0.229f / 255.f, 1.0 / 0.224f / 255.f, 1.0 / 0.225f / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals); // 像素范围0~255

    ncnn::Mat out;
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);           // 输入
    ex.extract("out0", out);       // 输出
    std::vector<float> cls_scores; // 所有分类的得分
    cls_scores.resize(out.w);      // 分类个数
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }
    print_topk(cls_scores, 5);
    return 0;
}