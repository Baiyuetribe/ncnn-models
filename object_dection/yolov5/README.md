# yolov5

## Input --> Output

![](https://github.com/ultralytics/yolov5/releases/download/v1.0/splash.jpg)

## Convert 

pt --> TorchScript --> pnnx --> ncnn
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


