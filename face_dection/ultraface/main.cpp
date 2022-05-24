//
//  main.cpp
//  UltraFaceTest
//
//  Created by vealocia on 2019/10/17.
//  Copyright © 2019 vealocia. All rights reserved.
//

#include "UltraFace.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
    UltraFace ultraface("models/version-RFB-320.bin", "models/version-RFB-320.param", 320, 240, 1, 0.7); // config model input
    std::string image_file = "intput.jpg";                                                               // input image
    cv::Mat frame = cv::imread(image_file);
    if (frame.empty())
    {
        std::cout << "read image failed" << std::endl;
        return -1;
    }
    ncnn::Mat inmat = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);

    std::vector<FaceInfo> face_info;
    ultraface.detect(inmat, face_info);
    for (int i = 0; i < face_info.size(); i++)
    {
        auto face = face_info[i];
        cv::Point pt1(face.x1, face.y1);
        cv::Point pt2(face.x2, face.y2);
        cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("UltraFace", frame);
    cv::waitKey();
    cv::imwrite("result.jpg", frame);
    return 0;
}
