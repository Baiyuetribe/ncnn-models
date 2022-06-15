git clone https://github.com/ultralytics/yolov5 # clone
cd yolov5
pip install -r requirements.txt # install
python export.py --weights yolov5s.pt --include torchscript --train
./pnnx yolov5s.torchscript inputshape=[1,3,640,640] inputshape2=[1,3,320,320]
# --train必须添加，否则会报错；inputshape2代表动态化尺寸，可以根据实际情况设置
