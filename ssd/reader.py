# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import xml.etree.ElementTree
import os
import time
import copy
import six
import math
import numpy as np
from PIL import Image
from PIL import ImageDraw
import image_util
import paddle


class Settings(object):
    def __init__(self,
                 dataset=None,
                 data_dir=None,
                 label_file=None,
                 resize_h=300,
                 resize_w=300,
                 mean_value=[127.5, 127.5, 127.5],
                 apply_distort=True,
                 apply_expand=True,
                 ap_version='11point'):
        self._dataset = dataset
        self._ap_version = ap_version
        self._data_dir = data_dir
        self._label_list = []
        label_fpath = os.path.join(data_dir, label_file)
        for line in open(label_fpath):
            self._label_list.append(line.strip())

        self._apply_distort = apply_distort
        self._apply_expand = apply_expand
        self._resize_height = resize_h
        self._resize_width = resize_w
        self._img_mean = np.array(mean_value)[:, np.newaxis, np.newaxis].astype(
            'float32')
        self._expand_prob = 0.5
        self._expand_max_ratio = 4
        self._hue_prob = 0.5
        self._hue_delta = 18
        self._contrast_prob = 0.5
        self._contrast_delta = 0.5
        self._saturation_prob = 0.5
        self._saturation_delta = 0.5
        self._brightness_prob = 0.5
        self._brightness_delta = 0.125

    @property
    def dataset(self):
        return self._dataset

    @property
    def ap_version(self):
        return self._ap_version

    @property
    def apply_distort(self):
        return self._apply_expand

    @property
    def apply_distort(self):
        return self._apply_distort

    @property
    def data_dir(self):
        return self._data_dir

    @data_dir.setter
    def data_dir(self, data_dir):
        self._data_dir = data_dir

    @property
    def label_list(self):
        return self._label_list

    @property
    def resize_h(self):
        return self._resize_height

    @property
    def resize_w(self):
        return self._resize_width

    @property
    def img_mean(self):
        return self._img_mean


def preprocess(img, bbox_labels, mode, settings):
    img_width, img_height = img.size
    sampled_labels = bbox_labels
    if mode == 'train':
        if settings._apply_distort:
            img = image_util.distort_image(img, settings)
        if settings._apply_expand:
            img, bbox_labels, img_width, img_height = image_util.expand_image(
                img, bbox_labels, img_width, img_height, settings)
        # sampling
        batch_sampler = []
        # hard-code here
        batch_sampler.append(
            image_util.sampler(1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0))
        batch_sampler.append(
            image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.1, 0.0))
        batch_sampler.append(
            image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.3, 0.0))
        batch_sampler.append(
            image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.5, 0.0))
        batch_sampler.append(
            image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.7, 0.0))
        batch_sampler.append(
            image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.9, 0.0))
        batch_sampler.append(
            image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.0, 1.0))
        sampled_bbox = image_util.generate_batch_samples(batch_sampler,
                                                         bbox_labels)

        img = np.array(img)
        if len(sampled_bbox) > 0:
            idx = int(np.random.uniform(0, len(sampled_bbox)))
            img, sampled_labels = image_util.crop_image(
                img, bbox_labels, sampled_bbox[idx], img_width, img_height)

        img = Image.fromarray(img)
    img = img.resize((settings.resize_w, settings.resize_h), Image.ANTIALIAS)
    img = np.array(img)

    if mode == 'train':
        mirror = int(np.random.uniform(0, 2))
        if mirror == 1:
            img = img[:, ::-1, :]
            for i in six.moves.xrange(len(sampled_labels)):
                tmp = sampled_labels[i][1]
                sampled_labels[i][1] = 1 - sampled_labels[i][3]
                sampled_labels[i][3] = 1 - tmp
    # HWC to CHW
    if len(img.shape) == 3:
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 1, 0)
    # RBG to BGR
    img = img[[2, 1, 0], :, :]
    img = img.astype('float32')
    img -= settings.img_mean
    img = img * 0.007843
    return img, sampled_labels


def Mydata(settings, file_list, mode, batch_size, shuffle):
    def reader():
        if mode == 'train' and shuffle:
            np.random.shuffle(file_list)
        batch_out = []
        cnt = 0
        for image in file_list:
            image_path, label_path = image.split()
            image_path = os.path.join(settings.data_dir, image_path)
            label_path = os.path.join(settings.data_dir, label_path)
            if not os.path.exists(image_path):
                raise ValueError("%s is not exist, you should specify "
                                 "data path correctly." % image_path)
            im = Image.open(image_path)
            if im.mode == 'L':
                im = im.convert('RGB')
            im_width, im_height = im.size

            # layout: label | xmin | ymin | xmax | ymax | difficult
            bbox_labels = []
            root = xml.etree.ElementTree.parse(label_path).getroot()
            for object in root.findall('object'):
                bbox_sample = []
                # start from 1
                bbox_sample.append(
                    float(settings.label_list.index(object.find('name').text)))
                bbox = object.find('bndbox')
                try:
                    difficult = float(object.find('difficult').text)
                except:
                    difficult = float(object.find('Difficult').text)
                bbox_sample.append(float(bbox.find('xmin').text) / im_width)
                bbox_sample.append(float(bbox.find('ymin').text) / im_height)
                bbox_sample.append(float(bbox.find('xmax').text) / im_width)
                bbox_sample.append(float(bbox.find('ymax').text) / im_height)
                bbox_sample.append(difficult)
                bbox_labels.append(bbox_sample)
            im, sample_labels = preprocess(im, bbox_labels, mode, settings)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) == 0: continue
            im = im.astype('float32')
            boxes = sample_labels[:, 1:5]
            lbls = sample_labels[:, 0].astype('int32')
            difficults = sample_labels[:, -1].astype('int32')

            batch_out.append((im, boxes, lbls, difficults))
            if len(batch_out) == batch_size:
                yield batch_out
                cnt += len(batch_out)
                batch_out = []

        if mode == 'test' and len(batch_out) > 1:
            yield batch_out
            cnt += len(batch_out)
            batch_out = []

    return reader

def train(settings,
          file_list,
          batch_size,
          shuffle=True,
          num_workers=8,
          enable_ce=False):
    file_path = os.path.join(settings.data_dir, file_list)
    readers = []
    images = [line.strip() for line in open(file_path)]
    np.random.shuffle(images)
    n = int(math.ceil(len(images) // num_workers))
    image_lists = [images[i:i + n] for i in range(0, len(images), n)]
    for l in image_lists:
        readers.append(Mydata(settings, l, 'train', batch_size, shuffle))
    return paddle.reader.multiprocess_reader(readers, False)


def test(settings, file_list, batch_size):
    file_list = os.path.join(settings.data_dir, file_list)
    image_list = [line.strip() for line in open(file_list)]
    return Mydata(settings, image_list, 'test', batch_size, False)


def infer(settings, image):
    def reader():
        img = copy.deepcopy(image)
        if img.mode == 'L':
            img = img.convert('RGB')
        im_width, im_height = img.size
        img = img.resize((settings.resize_w, settings.resize_h),
                         Image.ANTIALIAS)
        img = np.array(img)
        # HWC to CHW
        if len(img.shape) == 3:
            img = np.swapaxes(img, 1, 2)
            img = np.swapaxes(img, 1, 0)
        # RBG to BGR
        img = img[[2, 1, 0], :, :]
        img = img.astype('float32')
        img -= settings.img_mean
        img = img * 0.007843
        return img

    return reader
