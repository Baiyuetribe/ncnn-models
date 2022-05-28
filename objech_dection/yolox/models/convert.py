import os

# 0. 项目下载
os.system("git clone https://github.com/Megvii-BaseDetection/YOLOX.git")

# 1. 模型下载yolox_s,yolox_m,yolox_l,yolox_xl,yolox_Darknet53
os.system("wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth -O yolox_nano.pth")

# 2. pt ---> torchscript
os.system("python3 tools/export_torchscript.py --output-name ts.pt -n yolox-nano -c yolox_nano.pth")

# 3. ts ---> pnnx ---> ncnn
os.system("pnnx ts.pt inputshape=[1,3,416,416]")  # nano 和tiny输入尺寸为416*416.其余为640*640.

# 注意还有后处理，参见readme.md
