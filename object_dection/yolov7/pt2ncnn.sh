git clone https://github.com/WongKinYiu/yolov7.git # clone
cd yolov7
pip install -r requirements.txt # install
# download models
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
python models/export.py --weights yolov7 # export
pnnx yolov7.torchscript.pt inputshape=[1,3,640,640] inputshape=[1,3,320,320]
