# GPT2-Chinese

## Input --> Output

![](https://github.com/Morizeyao/GPT2-Chinese/raw/old_gpt_2_chinese_before_2021_4_22/sample/%E6%95%A3%E6%96%871.png)
![](https://github.com/EdVince/GPT2-ChineseChat-NCNN/raw/main/resources/android.jpg)

## Convert [❌]

pt --> TorchScript --> pnnx --> ncnnOptimize --> ncnn

```python
# 暂无方法，欢迎pr
```

## 复现结果

个人尝试复现，输出结果如下：
```log
  cpp.vcxproj -> C:\Users\baiyue\Desktop\Dev\桌面APP\sdk开发\gpt2-sdk\Release\cpp.exe
割
[0 NVIDIA GeForce RTX 2070 SUPER]  queueC=2[8]  queueG=0[16]  queueT=1[2]
[0 NVIDIA GeForce RTX 2070 SUPER]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[0 NVIDIA GeForce RTX 2070 SUPER]  fp16-p/s/a=1/1/1  int8-p/s/a=1/1/1
[0 NVIDIA GeForce RTX 2070 SUPER]  subgroup=32  basic=1  vote=1  ballot=1  shuffle=1
尽量用中文，目前英文有点小问题，输入quit退出，输入refresh清空记忆
user:你好
chatbot:[PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD]
user:
```
[PAD]为字典首字符，暂未清楚什么原因导致失败的，欢迎pr或issue讨论

## NCNN-MODELS

- [Download](https://github.com/Baiyuetribe/ncnn-models/releases/tag/models)

## Example project

- [Android: EdVince/GPT2-ChineseChat-NCNN](https://github.com/EdVince/GPT2-ChineseChat-NCNN)

## Reference

- [Morizeyao/GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese)
- [EdVince/GPT2-ChineseChat-NCNN](https://github.com/EdVince/GPT2-ChineseChat-NCNN)
- [Tencent/ncnn](https://github.com/Tencent/ncnn/blob/master/examples/rvm.cpp)


