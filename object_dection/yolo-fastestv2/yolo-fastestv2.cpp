// opencv2
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// ncnn
#include "net.h"
#include <iostream>
#include <vector>

class TargetBox
{
private:
    float getWidth() { return (x2 - x1); };
    float getHeight() { return (y2 - y1); };

public:
    int x1;
    int y1;
    int x2;
    int y2;

    int cate;
    float score;

    float area() { return getWidth() * getHeight(); };
};

//输出节点数
int numOutput = 2;
int inputWidth = 352; // 模型输入尺寸
int inputHeight = 352;
float thresh = 0.3;     // 置信度
int numAnchor = 3;      // anchor num
int numCategory = 80;   //类别数目
float nmsThresh = 0.25; // NMS阈值
// anchor box w h
std::vector<float> bias{12.64, 19.39, 37.88, 51.48, 55.71, 138.31,
                        126.91, 78.23, 131.57, 214.55, 279.92, 258.87};
std::vector<float> anchor;

//检测类别分数处理
int getCategory(const float *values, int index, int &category, float &score)
{
    float tmp = 0;
    float objScore = values[4 * numAnchor + index];

    for (int i = 0; i < numCategory; i++)
    {
        float clsScore = values[4 * numAnchor + numAnchor + i];
        clsScore *= objScore;

        if (clsScore > tmp)
        {
            score = clsScore;
            category = i;

            tmp = clsScore;
        }
    }

    return 0;
}

//特征图后处理
int predHandle(const ncnn::Mat *out, std::vector<TargetBox> &dstBoxes, const float scaleW, const float scaleH, const float thresh)
{ // do result
    anchor.assign(bias.begin(), bias.end());
    for (int i = 0; i < numOutput; i++)
    {
        int stride;
        int outW, outH, outC;

        outH = out[i].c;
        outW = out[i].h;
        outC = out[i].w;

        assert(inputHeight / outH == inputWidth / outW);
        stride = inputHeight / outH;

        for (int h = 0; h < outH; h++)
        {
            const float *values = out[i].channel(h);

            for (int w = 0; w < outW; w++)
            {
                for (int b = 0; b < numAnchor; b++)
                {
                    // float objScore = values[4 * numAnchor + b];
                    TargetBox tmpBox;
                    int category = -1;
                    float score = -1;

                    getCategory(values, b, category, score);

                    if (score > thresh)
                    {
                        float bcx, bcy, bw, bh;

                        bcx = ((values[b * 4 + 0] * 2. - 0.5) + w) * stride;
                        bcy = ((values[b * 4 + 1] * 2. - 0.5) + h) * stride;
                        bw = pow((values[b * 4 + 2] * 2.), 2) * anchor[(i * numAnchor * 2) + b * 2 + 0];
                        bh = pow((values[b * 4 + 3] * 2.), 2) * anchor[(i * numAnchor * 2) + b * 2 + 1];

                        tmpBox.x1 = (bcx - 0.5 * bw) * scaleW;
                        tmpBox.y1 = (bcy - 0.5 * bh) * scaleH;
                        tmpBox.x2 = (bcx + 0.5 * bw) * scaleW;
                        tmpBox.y2 = (bcy + 0.5 * bh) * scaleH;
                        tmpBox.score = score;
                        tmpBox.cate = category;

                        dstBoxes.push_back(tmpBox);
                    }
                }
                values += outC;
            }
        }
    }
    return 0;
}

bool scoreSort(TargetBox a, TargetBox b)
{
    return (a.score > b.score);
}
float intersection_area(const TargetBox &a, const TargetBox &b)
{
    if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

    return inter_width * inter_height;
}

// NMS处理
int nmsHandle(std::vector<TargetBox> &tmpBoxes, std::vector<TargetBox> &dstBoxes)
{
    std::vector<int> picked;

    sort(tmpBoxes.begin(), tmpBoxes.end(), scoreSort);

    for (int i = 0; i < tmpBoxes.size(); i++)
    {
        int keep = 1;
        for (int j = 0; j < picked.size(); j++)
        {
            //交集
            float inter_area = intersection_area(tmpBoxes[i], tmpBoxes[picked[j]]);
            //并集
            float union_area = tmpBoxes[i].area() + tmpBoxes[picked[j]].area() - inter_area;
            float IoU = inter_area / union_area;

            if (IoU > nmsThresh && tmpBoxes[i].cate == tmpBoxes[picked[j]].cate)
            {
                keep = 0;
                break;
            }
        }

        if (keep)
        {
            picked.push_back(i);
        }
    }

    for (int i = 0; i < picked.size(); i++)
    {
        dstBoxes.push_back(tmpBoxes[picked[i]]);
    }

    return 0;
}

int main()
{
    cv::Mat cvImg = cv::imread("input.png"); // 测试图片
    if (cvImg.empty())
    {
        std::cout << "read image failed" << std::endl;
        return -1;
    }
    ncnn::Net net;                                 // 推理定义
    net.load_param("models/yolo-fastestv2.param"); // yolo-fastestv2-opt
    net.load_model("models/yolo-fastestv2.bin");
    // 预处理
    float scaleW = (float)cvImg.cols / (float)inputWidth;
    float scaleH = (float)cvImg.rows / (float)inputHeight;
    // resize of input image data
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(cvImg.data, ncnn::Mat::PIXEL_BGR, cvImg.cols, cvImg.rows, inputWidth, inputHeight);
    // Normalization of input image data
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    // creat extractor
    ncnn::Extractor ex = net.create_extractor();
    ex.input("input.1", in); // 入口 3*352*352  // input.1
    // forward
    ncnn::Mat out[2];
    ex.extract("794", out[0]); // 95*22x22
    ex.extract("796", out[1]); // 95*11x11
    // 特征图后处理
    std::vector<TargetBox> boxes; // 结果集合
    std::vector<TargetBox> tmpBoxes;
    predHandle(out, tmpBoxes, scaleW, scaleH, thresh);
    // NMS
    nmsHandle(tmpBoxes, boxes);
    // 绘制结果
    static const char *class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"}; // 内置的一些分类
    for (int i = 0; i < boxes.size(); i++)
    {
        std::cout << boxes[i].x1 << " " << boxes[i].y1 << " " << boxes[i].x2 << " " << boxes[i].y2
                  << " " << boxes[i].score << " " << boxes[i].cate << std::endl;

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[boxes[i].cate], boxes[i].score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = boxes[i].x1;
        int y = boxes[i].y1 - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > cvImg.cols)
            x = cvImg.cols - label_size.width;

        cv::rectangle(cvImg, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(cvImg, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        cv::rectangle(cvImg, cv::Point(boxes[i].x1, boxes[i].y1),
                      cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 255, 0), 2, 2, 0);
    }

    // cv::imwrite("output.png", cvImg);
    cv::imshow("result", cvImg);
    cv::waitKey(0);
    return 0;
}
