# yolox

## Input --> Output

![](https://github.com/Megvii-BaseDetection/YOLOX/raw/main/assets/git_fig.png)

## Convert 

pt --> TorchScript --> pnnx  --> ncnn

```python
import os

# 0. 项目下载
os.system("git clone https://github.com/Megvii-BaseDetection/YOLOX.git")

# 1. 模型下载yolox_tiny,yolox_s,yolox_m,yolox_l,yolox_xl,yolox_Darknet53
os.system("wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth -O yolox_nano.pth")

# 2. pt ---> torchscript
os.system("python3 tools/export_torchscript.py --output-name ts.pt -n yolox-nano -c yolox_nano.pth")

# 3. ts ---> pnnx ---> ncnn
os.system("pnnx ts.pt inputshape=[1,3,416,416]")  # nano 和tiny输入尺寸为416*416.其余为640*640.
```
生成的param文件里，手动移除Input后面到Contact6层，然后新加一层，最后变为如下效果：
```ruby
7767517
279 317
Input                    in0                      0 1 in0
YoloV5Focus              focus                    1 1 in0 9
```

## IsWork?

运行后，有结果输出，但后处理无反应

## Example project

- [Android: FeiGeChuanShu/ncnn-android-yolox](https://github.com/FeiGeChuanShu/ncnn-android-yolox)
  
## Reference

- [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)


