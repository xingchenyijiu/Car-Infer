import cv2
from PIL import Image
import infer1
f=infer1.getF()
video_path=''
cap=cv2.VideoCapture(video_path)
while True:
    ret,frame=cap.read()
    if ret:
        xmin,ymin,xmax,ymax,label=f(Image.fromarray(frame))
    else:
        break
cap.release()
