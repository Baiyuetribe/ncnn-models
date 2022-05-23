# RVM

## Input --> Output

![](https://github.com/PeterL1n/RobustVideoMatting/raw/master/documentation/image/showreel.gif)

## Convert [❌]

pt --> TorchScript --> pnnx --> ncnnOptimize --> ncnn

```python
import os
import torch
# 0. pt模型下载及初始化
model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")  # or "resnet50"
model.cpu()
model.eval()
# model(src, *rec, downsample_ratio=0.25) # src can be [B, C, H, W] or [B, T, C, H, W] RGB input is normalized to 0~1 range.
# 1. pt --> torchscript
traced_script_module = torch.jit.trace(model, torch.randn(1, 3, 512, 512), strict=False)
traced_script_module.save("ts.pt")

# 2. ts --> pnnx --> ncnn
os.system("pnnx ts.pt inputshape=[1,3,512,512],[1,3,320,320]  device=cpu")    # 可能错误

# 3. ncnn ---> optmize ----> ncnn
os.system("ncnnoptimize ts.ncnn.param ts.ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16
```
报错内容如下：
```log
# 运行后c++报错：原因暂时不清楚
# pass_level5
# pass_ncnn
# ignore pnnx.Expression pnnx_expr_256 param expr=1
# create_custom_layer pnnx.Expression
# fuse_convolution_activation conv_12 hswish_83
```
c++运行后报错：
```
layer pnnx_expr_256 not exists or registered
```

## Example project

- [Desktop: RVM-GUI](https://github.com/Baiyuetribe/paper2gui/blob/main/ImageMatting/rvm_gui.md)
- [Android: FeiGeChuanShu/ncnn_Android_RobustVideoMatting](https://github.com/FeiGeChuanShu/ncnn_Android_RobustVideoMatting)

## Reference

- [Tencent/ncnn](https://github.com/Tencent/ncnn/blob/master/examples/rvm.cpp)
- [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)


