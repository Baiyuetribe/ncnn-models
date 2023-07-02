# ROOP 换脸

## Input --> Output

![](https://github.com/s0md3v/roop/blob/main/demo.gif)

## 出入信息
### 输入尺寸
target [1,3,128,128]
source [1,512]
### 输出尺寸
output [1,3,128,128]

## 模型转换


```python

## 1. 下载onnx模型
wget https://huggingface.co/optobsafetens/inswapper_128/resolve/main/roopVideoFace_v10.onnx

## 2. onnxsim简化
onnxsim in.onnx sim.onnx

## 3. onnx2ncnn
onnx2ncnn sim.onnx sim.param sim.bin

## 4. fp16量化
ncnnoptimize sim.param sim.bin opt.param opt.bin 1

## 5. 模型信息
Op.           Total  arm  loongarch  mips  riscv  vulkan  x86  
BinaryOp      92     Y    Y          Y     Y      Y       Y    
Split         43                                               
Reduction     24                                               
Crop          24     Y    Y          Y     Y      Y       Y    
Convolution   20     Y    Y          Y     Y      Y       Y    
Padding       14     Y    Y          Y     Y      Y       Y    
UnaryOp       13     Y    Y          Y     Y      Y       Y    
InnerProduct  12     Y    Y          Y     Y      Y       Y    
ExpandDims    12                                               
ReLU          6      Y    Y          Y     Y      Y       Y    
Interp        2      Y    Y          Y     Y      Y       Y  

```

## python实现
https://github.com/s0md3v/sd-webui-roop/blob/main/scripts/faceswap.py

## c++实现

欢迎pr

## Example project



## Reference

- [s0md3v/roop](https://github.com/s0md3v/roop)
- [Tencent/ncnn](https://github.com/Tencent/ncnn)


