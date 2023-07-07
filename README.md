## <p align="center"> NCNN Models </p>

The collection of pre-trained AI models, and how they were converted, deployed. [中文](README-CN.md)

![](docs/images/logo.png)

### About

The ncnn framework enables cross-device deployment with the help of the vulkan api. We pre-train models via pytorch, tensorflow, paddle etc. and then convert them to ncnn models for final deployment on Windows, mac, linux, android, ios, WebAssembly and uni-app. However, model conversion is not a one-click process and needs to be handled manually. In order to extend the boundary applications of ncnn, we have created this repository to receive any cases of successful or failed conversions.

### How to contribute

[Contribute tutorial](contribute.md)

✅ : good to work
❌ : bad to work
⭕ : good to work, but not good to contribute
🤔 : not sure, but good to contribute
🔥🔥💥

### Ncnn Models

> We believe we will succeed in the end. 😂。

| Model                                             | Year | Size   | From    | Type                      | Convert | NCNN | Hot |
| :------------------------------------------------ | :--- | :----- | :------ | :------------------------ | :------ | :--- | :-- |
| [roop](face_swap/roop)                            | 2023 | 276.7M | Onnx    | face_swap                 | ✅      | 🤔   | 🔥  |
| [nerf](video/nerf)                                | 2023 | 0.1MB  | Pytorch | instant-ngp               | ✅      | ✅   |     |
| [codeformer](face_restoration/codeformer)         | 2023 | 212.5M | Pytorch | face_restoration          | ✅      | ✅   | 🔥  |
| [vits](tts/vits)                                  | 2022 | 91MB   | Pytorch | tts                       | ✅      | ✅   | 🔥  |
| [stablediffusion](diffusion/stablediffuson)       | 2022 | 1.7GB  | Pytorch | diffusion                 | ✅      | ✅   | 🔥  |
| [sherpa](tts/sherpa)                              | 2022 | 134MB  | Pytorch | tts                       | ✅      | ✅   | 🔥  |
| [DTLN](audio_denoising/dtln)                      | 2022 | 1.9MB  | Pytorch | audio_denoising           | ✅      | ✅   |     |
| [gpt2-chinese](nlp/gpt2-chinese)                  | 2022 | 357MB  |         | nlp                       |         | ⭕   |     |
| [MAT](image_inpainting/mat)                       | 2022 |        | Pytorch | image_inpainting          | ❌      |      |     |
| [RVM](image_matting/RVM)                          | 2021 | 13.6MB | Pytorch | image_matting             |         | ✅   |     |
| [vitea](image_matting/vitea)                      | 2022 | 52.9MB | Pytorch | image_matting             | ❌      |      |     |
| [AnimeGanV3](style_transfer/animeganv3)           | 2022 |        | Onnx    | style_transfer            | ❌      |      |     |
| [HybridNets](object_dection/hybridnets)           | 2022 |        | Pytorch | object_dection            | ❌      |      |     |
| [yolop](object_dection/yolop)                     | 2021 |        | Pytorch | object_dection            | ❌      | 🤔   | 💥  |
| [pfld](face_dection/pfld)                         | 2019 | 4.9MB  | Pytorch | face_dection              | ❌      | ✅   |     |
| [Anime](face_dection/Anime_Face)                  | 2021 | 18.8MB | Onnx    | face_dection              | ✅      | ⭕   |     |
| [CaiT](image_classification/cait)                 | 2021 | 34.3MB | Pytorch | image_classification      | ✅      |      |     |
| [FastestDet](object_dection/fastestdet)           | 2022 | 0.4MB  | Pytorch | object_dection            | ✅      | ✅   | 💥  |
| [yolov7](object_dection/yolov7)                   | 2022 | 12.1MB | Pytorch | object_dection            | ✅      | ✅   |     |
| [yolov6](object_dection/yolov6)                   | 2022 | 8.4MB  | Pytorch | object_dection            | ⭕      | ✅   |     |
| [yolov5](object_dection/yolov5)                   | 2021 | 2.3MB  | Pytorch | object_dection            | ✅      | ✅   | 💥  |
| [yolo-fastestv2](object_dection/yolo-fastestv2)   | 2021 | 0.4MB  | Pytorch | object_dection            | ✅      | ✅   | 💥  |
| [yolox](object_dection/yolox)                     | 2021 | 1.7MB  | Pytorch | object_dection            | ✅      | ✅   |     |
| [nanodet](object_dection/nanodet)                 | 2020 | 2.3MB  | Onnx    | object_dection            | ✅      | ✅   |     |
| [DenseNet](image_classification/denseNet)         | 2018 | 21.5MB | Pytorch | image_classification      | ✅      | ✅   |     |
| [resnet18](image_classification/resnet18)         | 2015 | 22.8MB | Pytorch | image_classification      | ✅      | ✅   |     |
| [mobilenet_v2](image_classification/mobilenet_v2) | 2019 | 6.8MB  | Pytorch | image_classification      | ✅      | ✅   |     |
| [mobilenet_v3](image_classification/mobilenet_v3) | 2019 | 10.7MB | Pytorch | image_classification      | ✅      | ✅   |     |
| [Res2Net](image_classification/res2net)           | 2021 | 88.2MB | Pytorch | image_classification      | ✅      | ✅   |     |
| [Res2Next50](image_classification/res2next50)     | 2021 | 48.1MB | Pytorch | image_classification      | ✅      | ✅   |     |
| [shufflenetv2](image_classification/shufflenetv2) | 2018 | 4.4MB  | Onnx    | image_classification      | ✅      | ✅   |     |
| [vgg16](image_classification/vgg16)               | 2015 | 263MB  | Pytorch | image_classification      | ✅      | ✅   |     |
| [efficientnet](image_classification/efficientnet) | 2021 | 10.3MB | Pytorch | image_classification      | ✅      | ✅   |     |
| [deeplabv3](image_matting/deeplabv3)              | 2017 | 21.5MB | Pytorch | image_matting             | ✅      | ✅   |     |
| [yolov7-mask](image_matting/yolov7_mask)          | 2022 | 86.6MB | Pytorch | image_matting             | 🤔      | ✅   |     |
| [deoldify](image_inpainting/deoldify)             | 2019 | 242MB  | Onnx    | image_inpainting          | 🤔      | ✅   |     |
| [UltraFace](face_dection/ultraface)               | 2019 | 0.6MB  | Pytorch | face_dection              | ✅      | ✅   |     |
| [Anime2Real](style_transfer/anime2real)           | 2022 | 22.2MB | Pytorch | style_transfer            | ✅      | ✅   |     |
| [AnimeGanV2](style_transfer/animeganv2)           | 2020 | 4.2MB  | Pytorch | style_transfer            | ✅      | ✅   |     |
| [styletransfer](style_transfer/styletransfer)     | 2016 | 3.2MB  | Onnx    | style_transfer            | ✅      | ✅   |     |
| [ifrnet](video/ifrnet)                            | 2022 | 5.6MB  | Pytorch | video_frame_interpolation |         | ✅   |     |
| [Rife](video/rife)                                | 2021 | 10MB   | Onnx    | video_frame_interpolation |         | ✅   |     |
| [GFPGAN](face_restoration/GFPGAN)                 | 2021 | 214MB  | Onnx    | face_restoration          |         | ✅   | 💥  |

### Awesome App based on Ncnn

#### 1. Deep Face Live

![](https://github.com/gunpowder78/DeepFaceLive/raw/master/doc/deepfacelive_intro.png)

see [DeepFaceLive](https://github.com/gunpowder78/DeepFaceLive)

#### 2. video-super-resolution

waifu2x-ncnn-vulkan、realcugan-ncnn-vulkan、realEsrgan-ncnn-vulkan ...
![](https://github.com/Baiyuetribe/paper2gui/raw/main/docs/images/realESRGAN_RAM.png)

see [RealESRGAN](https://github.com/Baiyuetribe/paper2gui/blob/main/Video%20Super%20Resolution/RealESRGAN-GUI-RAM.md)

#### 3. Video Matting

![](https://github.com/ZHKKKe/MODNet/raw/master/doc/gif/homepage_demo.gif)

see [MODNet](https://github.com/Baiyuetribe/paper2gui/blob/main/VideoMatting/modnet_gui.md)

### 4. BlazePose

![](https://github.com/FeiGeChuanShu/ncnn_Android_BlazePose/raw/main/result.gif)
![](https://github.com/FeiGeChuanShu/ncnn_Android_BlazePose/raw/main/result_smoothing.gif)

see [BlazePose](https://github.com/FeiGeChuanShu/ncnn_Android_BlazePose)

### 5. AnimeGanV2

![](https://user-images.githubusercontent.com/26464535/142294796-54394a4a-a566-47a1-b9ab-4e715b901442.gif)

see [AnimeGanV2](https://github.com/Baiyuetribe/paper2gui/blob/main/Style%20Transfer/animegan_gui.md)

### 6. GPT2-ChineseChat-NCNN

![](https://github.com/EdVince/GPT2-ChineseChat-NCNN/raw/main/resources/android.jpg)

see [GPT2-ChineseChat-NCNN](https://github.com/EdVince/GPT2-ChineseChat-NCNN)

### QQ 群

- 824562395 【_加群请备注你正在转换的新模型(2022 至今)_】

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Baiyuetribe/ncnn-models&type=Date)](https://star-history.com/#Baiyuetribe/ncnn-models&Date)
