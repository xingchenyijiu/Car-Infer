import infer1
from PIL import Image
f=infer1.getF()
for i in range(1):
   	print(f(Image.open(str(i+1)+'.jpg')))
