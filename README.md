##  <p align="center"> ncnn models </p>

The collection of pre-trained AI models, and how they were converted, deployed. [中文](README-CN.md)

### About

The ncnn framework enables cross-device deployment with the help of the vulkan api. We pre-train models via pytorch, tensorflow, flying paddle etc. and then convert them to ncnn models for final deployment on Windows, mac, linux, android, ios, WebAssembly and uni-app. However, model conversion is not a one-click process and needs to be handled manually. In order to extend the boundary applications of ncnn, we have created this repository to receive any cases of successful or failed conversions.

### How to contribute

[Contribute tutorial](contribute.md)

	✅ : good to work
    ❌ : bad to work
    ⭕ : good to work, but not good to contribute
    ❓ : not sure
    🤔 : not sure, but good to contribute
    🤷 : not sure, but bad to contribute
    🤯 : not sure, but not good to contribute
### Failure Notes

> We believe we will succeed in the end.

| Model                                   | From    | Code                                                   | Convert | IsWork | Desktop | Mobile | Wasm | Uni-app | loT  |
| :-------------------------------------- | :------ | :----------------------------------------------------- | :------ | :----- | :------ | :----- | :--- | :------ | :--- |
| [AnimeGanV3](style_transfer/animeganv3) | Onnx    | [link](https://github.com/TachibanaYoshino/AnimeGANv3) | ❌       | ❌      | ❌       | ❌      | ❌    | ❌       | ❌    |
| [yolov5](objech_dection/yolov5)         | Pytorch | [link](https://github.com/bryandlee/animegan2-pytorch) | ⭕       | ✅      | ✅       | ✅      | ✅    | ✅       | ✅    |
| [AnimeGanV2](style_transfer/animeganv2) | Pytorch | [link](https://github.com/bryandlee/animegan2-pytorch) | ✅       | ✅      | ✅       | ✅      | 🤔    | 🤔       | ⭕    |


### Action recognition

> need to be contribute

### Background removal

### Face detection

### Frame Interpolation

### Image classification

### Image segmentation

### Object detection

### Object tracking

### Style transfer


### Super resolution

### Text recognition

