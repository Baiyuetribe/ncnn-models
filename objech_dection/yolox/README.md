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
结尾也是一个Contact层，补加一层Permute.
改前：
```ruby
Concat                   cat_17                   3 1 313 314 315 out0 0=1
```
改后：
```ruby
Concat                   cat_17                   3 1 313 314 315 316 0=1
Permute                  Transpose_333            1 1 316 out0 0=1
```
为什么加这一层？暂不清楚。不加之前输出out0.shape: 8400 85 1 2 原始加上Permute后结果是：85 8400 1 2。
也许不需要这一层处理，但要变更下后处理，个人尝试简单替换w、h是无效的。

## Example project

- [Android: FeiGeChuanShu/ncnn-android-yolox](https://github.com/FeiGeChuanShu/ncnn-android-yolox)
  
## Reference

- [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)


