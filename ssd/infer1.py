import os
import time
import numpy as np
import argparse
import functools
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import paddle
import paddle.fluid as fluid
import reader
from mobilenet_ssd import build_mobilenet_ssd
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('dataset',          str,   'Mydata',    "coco and pascalvoc.")
add_arg('use_gpu',          bool,  False,      "Whether use GPU.")
add_arg('image_path',       str,   '',        "The image used to inference and visualize.")
add_arg('model_dir',        str,   './model/best_model',     "The model path.")
add_arg('nms_threshold',    float, 0.45,   "NMS threshold.")
add_arg('confs_threshold',  float, 0.5,    "Confidence threshold to draw bbox.")
add_arg('resize_h',         int,   300,    "The resized image height.")
add_arg('resize_w',         int,   300,    "The resized image height.")
add_arg('mean_value_B',     float, 127.5,  "Mean value for B channel which will be subtracted.")  #123.68
add_arg('mean_value_G',     float, 127.5,  "Mean value for G channel which will be subtracted.")  #116.78
add_arg('mean_value_R',     float, 127.5,  "Mean value for R channel which will be subtracted.")  #103.94
# yapf: enable


def infer(args, data_args, model_dir):
    image_shape = [3, data_args.resize_h, data_args.resize_w]
    num_classes = 7
    label_list = data_args.label_list

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    locs, confs, box, box_var = build_mobilenet_ssd(image, num_classes,
                                                    image_shape)
    nmsed_out = fluid.layers.detection_output(
        locs, confs, box, box_var, nms_threshold=args.nms_threshold)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if model_dir:
        def if_exist(var):
            return os.path.exists(os.path.join(model_dir, var.name))
        fluid.io.load_vars(exe, model_dir, predicate=if_exist)
    # yapf: enable
    #infer_reader = reader.infer(data_args, image_path)
    feeder = fluid.DataFeeder(place=place, feed_list=[image])

    #data = infer_reader()

    # switch network to test mode (i.e. batch norm test mode)
    test_program = fluid.default_main_program().clone(for_test=True)
    #print('################')
    def work(img):
        #nonlocal reader
        nonlocal feeder
        nonlocal test_program
        nonlocal data_args
        nonlocal nmsed_out
        nonlocal exe
        nonlocal args
        nonlocal label_list

        data=reader.infer(data_args,img)()
        nmsed_out_v, = exe.run(test_program,
                               feed=feeder.feed([[data]]),
                               fetch_list=[nmsed_out],
                               return_numpy=False)
        nmsed_out_v = np.array(nmsed_out_v)
       # draw_bounding_box_on_image(img, nmsed_out_v, args.confs_threshold, label_list)
        return getList(img, nmsed_out_v,args.confs_threshold, label_list)
    return work

def getList(image, nms_out, confs_threshold, label_list):
    im_width, im_height=image.size
    rxmin,rymin,rxmax,rymax,rlabel=[],[],[],[],[]

    for dt in nms_out:
        if dt[1]<confs_threshold:
            continue
        rlabel.append(label_list[int(dt[0])])
        
        xmin,ymin,xmax,ymax=clip_bbox(dt[2:])
        rxmin.append(xmin*im_width)
        rymin.append(ymin*im_height)
        rxmax.append(xmax*im_width)
        rymax.append(ymax*im_height)
    return rxmin,rymin,rxmax,rymax,rlabel


def draw_bounding_box_on_image(image, nms_out, confs_threshold,
                               label_list):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size

    for dt in nms_out:
        if dt[1] < confs_threshold:
            continue
        category_id = dt[0]
        bbox = dt[2:]
        xmin, ymin, xmax, ymax = clip_bbox(dt[2:])
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        draw.line(
            [(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=4,
            fill='green')
        if image.mode == 'RGB':
            draw.text((left, top), label_list[int(category_id)], (0, 255, 0))
    image_name = "images/19.jpg" 
    print("image with bbox drawed saved as {}".format(image_name))
    image.save(image_name)
    print('-----')


def clip_bbox(bbox):
    xmin = max(min(bbox[0], 1.), 0.)
    ymin = max(min(bbox[1], 1.), 0.)
    xmax = max(min(bbox[2], 1.), 0.)
    ymax = max(min(bbox[3], 1.), 0.)
    return xmin, ymin, xmax, ymax


def getF():
	#模型加载
    args = parser.parse_args()
    print_arguments(args)

    data_dir = 'data/Mydata'
    label_file = 'label_list'

    data_args = reader.Settings(
        dataset=args.dataset,
        data_dir=data_dir,
        label_file=label_file,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        mean_value=[args.mean_value_B, args.mean_value_G, args.mean_value_R],
        apply_distort=False,
        apply_expand=False,
        ap_version='')
    f=infer(
        args,
        data_args=data_args,
        model_dir=args.model_dir)
    return f
