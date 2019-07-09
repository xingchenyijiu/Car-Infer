from PIL import Image
import numpy as np
path=input()
img=Image.open(path)
if img.mode == 'L':
	img=img.convert('RGB')
w,h=img.size
img=img.resize((300,300),Image.ANTIALIAS)
img=np.array(img)
if len(img.shape)==3:
	img=np.swapaxes(img,1,2)
	img=np.swapaxes(img,1,0)
img=img[[2,1,0],:,:]
img=img.astype('float32')
img=img-127.5
img=img*0.007843
with open('mat.txt','w') as f:
	for i in range(3):
		for j in range(300):
			for k in range(300):
				f.write(str(img[i][j][k])+'\n')
