# YOLOv6

## Input --> Output

![](https://github.com/meituan/YOLOv6/raw/main/assets/picture.png)

## Convert 

pt --> TorchScript --> pnnx  --> ncnn

```python
# import os

# # 0. 项目下载
# os.system("git clone https://github.com/meituan/YOLOv6.git")

# # 1. 模型下载yolox_tiny,yolox_s,yolox_m,yolox_l,yolox_xl,yolox_Darknet53

# # 2. pt ---> torchscript
# os.system("python3 tools/export_torchscript.py --output-name ts.pt -n yolox-nano -c yolox_nano.pth")

# # 3. ts ---> pnnx ---> ncnn
# os.system("pnnx ts.pt inputshape=[1,3,416,416] inputshape2=[1,3,640,640]")  # nano 和tiny输入尺寸为416*416.其余为640*640.
```
通过onnx和PNNX转换均存在以下胶水op,暂无详细手动调整操作

## NCNN-MODELS

- [Download](https://github.com/Baiyuetribe/ncnn-models/releases/tag/models)

## Example project

- [Android: FeiGeChuanShu/ncnn-android-yolov6](https://github.com/FeiGeChuanShu/ncnn-android-yolov6)
  
## Reference

- [meituan/YOLOv6](https://github.com/meituan/YOLOv6)


