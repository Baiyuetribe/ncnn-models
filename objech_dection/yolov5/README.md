# yolov5

## Input --> Output

![](https://github.com/ultralytics/yolov5/releases/download/v1.0/splash.jpg)

## Convert 

pt --> TorchScript --> pnnx --> ncnnOptimize --> ncnn

```python
import os
import torch
# 0. pt模型下载及初始化
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
model.train()  # -train
model.cpu()
# model.eval() --train

# 1. pt --> torchscript
traced_script_module = torch.jit.trace(model, torch.randn(1, 3, 640, 640))
traced_script_module.save("ts.pt")

# 2. ts --> pnnx --> ncnn
os.system("pnnx ts.pt inputshape=[1,3,640,640] inputshape2=[1,3,320,320] device=cpu")

# 3. ncnn ---> optmize ----> ncnn
os.system("ncnnoptimize ts.ncnn.param ts.ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16
```
报错内容如下：
```log
############# pass_level1
no attribute value
unknown Parameter value kind prim::Constant
no attribute value
no attribute value
no attribute value
############# pass_level2
############# pass_level3
assign unique operator name pnnx_unique_0 to model.model.model.0.conv
assign unique operator name pnnx_unique_1 to model.model.model.9.m
assign unique operator name pnnx_unique_2 to model.model.model.9.m
############# pass_level4
############# pass_level5
############# pass_ncnn
insert_reshape_pooling 4
insert_reshape_pooling 4
insert_reshape_pooling 4
create_custom_layer aten::type_as
model has custom layer, shape_inference skipped
model has custom layer, estimate_memory_footprint skipped
```
c++运行后报错：
```
layer aten::type_as not exists or registered
```

成功案例：
```
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
python export.py --weights yolov5s.pt --include torchscript --train
./pnnx yolov5s.torchscript inputshape=[1,3,640,640] inputshape2=[1,3,320,320]
```
## Example project

- [Android: FeiGeChuanShu/ncnn-android-yolox](https://github.com/FeiGeChuanShu/ncnn-android-yolox)
- [WASM: nihui/ncnn-webassembly-yolov5](https://github.com/nihui/ncnn-webassembly-yolov5)
- [Uni-app: 670***@qq.com](https://ext.dcloud.net.cn/plugin?id=5243)
  
## Reference

- [详细记录u版YOLOv5目标检测ncnn实现（第二版）](https://zhuanlan.zhihu.com/p/471357671)
- [ultralytics/yolov5](https://github.com/ultralytics/yolov5)


