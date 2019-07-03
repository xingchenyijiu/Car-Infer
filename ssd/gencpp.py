import os
import paddle
import paddle.fluid as fluid
from mobilenet_ssd import build_mobilenet_ssd

image_shape=[3,300,300]
num_classes=7
label_list='./data/Mydata/label_list'
model_dir='./model/best_model'

image=fluid.layers.data(name='img',shape=image_shape,dtype='float32')
locs,confs,box,box_var=build_mobilenet_ssd(image,num_classes,image_shape)
out=fluid.layers.detection_output(locs,confs,box,box_var,nms_threshold=0.45)
place=fluid.CPUPlace()
exe=fluid.Executor(place)

def if_exist(var):
	return os.path.exists(os.path.join(model_dir,var.name))
fluid.io.load_vars(exe,model_dir,predicate=if_exist)

fluid.io.save_inference_model(dirname='./cppmodel',feeded_var_names=['img'],target_vars=[out],executor=exe,model_filename='model',params_filename='param',program_only=False)

