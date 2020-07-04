
# coding: utf-8

# ## 一、模型训练
# ### 1.1 准备数据和模型
# #### 1.1.1 解压`coco`数据集和`baseline_model`基线模型

# In[1]:


# 解压数据集
# !cd /home/aistudio/data/data7122 && unzip -qo train2017.zip -d /home/aistudio/work/coco2017/
# !cd /home/aistudio/data/data7122 && unzip -qo test2017.zip -d /home/aistudio/work/coco2017/
# !cd /home/aistudio/data/data7122 && unzip -qo val2017.zip -d /home/aistudio/work/coco2017/
# !cd /home/aistudio/data/data7122 && unzip -qo image_info_test2017.zip -d /home/aistudio/work/coco2017/
# !cd /home/aistudio/data/data7122 && unzip -qo annotations_trainval2017.zip -d /home/aistudio/work/coco2017/
# !cd /home/aistudio/data/data7122 && unzip -qo PaddlePaddle_baseline_model.zip -d /home/aistudio/work/
# !cd work/coco2017/train2017 && ls |wc -l


# #### 1.1.2 下载ssd_mobilenet_v1_coco预训练模型

# In[2]:


# Download the pretrained_model.
# !echo "Downloading pretrained_model ..."
# !cd /home/aistudio/work/pretrained_model && wget http://paddlemodels.bj.bcebos.com/ssd_mobilenet_v1_coco.tar.gz
# !echo "Extractint..."
# !cd /home/aistudio/work/pretrained_model && tar -xf ssd_mobilenet_v1_coco.tar.gz


# ### 1.2 下载coco数据集python api 依赖包

# In[3]:


#安装coco数据集工具包
get_ipython().system('pip install pycocotools')


# ### 1.3 构建代码训练模型
# #### 1.3.1 导入依赖

# In[4]:


#coding:utf-8
#导入依赖
import os
import time
import numpy as np
import functools
import shutil
import math
import multiprocessing

#加载参数
def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# 这些flags需要在`import paddle`之前启用
set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # enable GC to save memory
)

import paddle
import paddle.fluid as fluid
import numpy as np


# #### 1.3.2 设置全局变量参数

# In[5]:


#coding:utf-8
#全部变量参数
g_learning_rate = 0.001
# g_learning_rate = 0.00007
g_batch_size = 64
# g_batch_size = 32
g_epoc_num = 120
g_use_gpu = True
#是否并行运行
g_parallel = True
#数据集名称：coco2017
g_dataset = 'coco2017'
g_model_save_dir = '/home/aistudio/work/models/mobilenet_ssd.model'
g_score_threshold = 0.005
g_nms_topk = 400
g_nms_posk = 100
g_nms_threshold = 0.45
#预训练模型
# g_pretrained_model = '/home/aistudio/work/pretrained_model/ssd_mobilenet_v1_coco/'
g_pretrained_model = '/home/aistudio/work/models/mobilenet_ssd.model/34_best/'
# g_pretrained_model = ''
g_ap_version = '11point'
#输入图片的shape
g_image_shape = '3,300,300'
#将被减去的B,G,R通道的平均值
g_mean_BGR = '127.5,127.5,127.5'
#数据源文件夹
g_data_dir = '/home/aistudio/work/coco2017/'
#是否使用多进程
g_use_multiprocess = True


# #### 1.3.3 设置数据集变量参数

# In[6]:


train_parameters = {
    "coco2017": {
        "train_images": 118287,
        "image_shape": [3, 300, 300],
        # "class_num": 91,
        "class_num": 81,
        "batch_size": 64,
        "lr": 0.001,
        "lr_epochs": [12, 19],
        "lr_decay": [1, 0.5, 0.25],
        "ap_version": 'integral', # should use eval_coco_map.py to test model
    }
}


# #### 1.3.4 图像预处理工具`image_util`

# In[7]:


#image_util
from PIL import Image, ImageEnhance, ImageDraw
from PIL import ImageFile
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True  #otherwise IOError raised image file is truncated


class sampler():
    def __init__(self, max_sample, max_trial, min_scale, max_scale,
                 min_aspect_ratio, max_aspect_ratio, min_jaccard_overlap,
                 max_jaccard_overlap):
        self.max_sample = max_sample
        self.max_trial = max_trial
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_jaccard_overlap = min_jaccard_overlap
        self.max_jaccard_overlap = max_jaccard_overlap


class bbox():
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def bbox_area(src_bbox):
    width = src_bbox.xmax - src_bbox.xmin
    height = src_bbox.ymax - src_bbox.ymin
    return width * height


def generate_sample(sampler):
    scale = np.random.uniform(sampler.min_scale, sampler.max_scale)
    aspect_ratio = np.random.uniform(sampler.min_aspect_ratio,
                                  sampler.max_aspect_ratio)
    aspect_ratio = max(aspect_ratio, (scale**2.0))
    aspect_ratio = min(aspect_ratio, 1 / (scale**2.0))

    bbox_width = scale * (aspect_ratio**0.5)
    bbox_height = scale / (aspect_ratio**0.5)
    xmin_bound = 1 - bbox_width
    ymin_bound = 1 - bbox_height
    xmin = np.random.uniform(0, xmin_bound)
    ymin = np.random.uniform(0, ymin_bound)
    xmax = xmin + bbox_width
    ymax = ymin + bbox_height
    sampled_bbox = bbox(xmin, ymin, xmax, ymax)
    return sampled_bbox


def jaccard_overlap(sample_bbox, object_bbox):
    if sample_bbox.xmin >= object_bbox.xmax or             sample_bbox.xmax <= object_bbox.xmin or             sample_bbox.ymin >= object_bbox.ymax or             sample_bbox.ymax <= object_bbox.ymin:
        return 0
    intersect_xmin = max(sample_bbox.xmin, object_bbox.xmin)
    intersect_ymin = max(sample_bbox.ymin, object_bbox.ymin)
    intersect_xmax = min(sample_bbox.xmax, object_bbox.xmax)
    intersect_ymax = min(sample_bbox.ymax, object_bbox.ymax)
    intersect_size = (intersect_xmax - intersect_xmin) * (
        intersect_ymax - intersect_ymin)
    sample_bbox_size = bbox_area(sample_bbox)
    object_bbox_size = bbox_area(object_bbox)
    overlap = intersect_size / (
        sample_bbox_size + object_bbox_size - intersect_size)
    return overlap


def satisfy_sample_constraint(sampler, sample_bbox, bbox_labels):
    if sampler.min_jaccard_overlap == 0 and sampler.max_jaccard_overlap == 0:
        return True
    for i in range(len(bbox_labels)):
        object_bbox = bbox(bbox_labels[i][1], bbox_labels[i][2],
                           bbox_labels[i][3], bbox_labels[i][4])
        overlap = jaccard_overlap(sample_bbox, object_bbox)
        if sampler.min_jaccard_overlap != 0 and                 overlap < sampler.min_jaccard_overlap:
            continue
        if sampler.max_jaccard_overlap != 0 and                 overlap > sampler.max_jaccard_overlap:
            continue
        return True
    return False


def generate_batch_samples(batch_sampler, bbox_labels):
    sampled_bbox = []
    index = []
    c = 0
    for sampler in batch_sampler:
        found = 0
        for i in range(sampler.max_trial):
            if found >= sampler.max_sample:
                break
            sample_bbox = generate_sample(sampler)
            if satisfy_sample_constraint(sampler, sample_bbox, bbox_labels):
                sampled_bbox.append(sample_bbox)
                found = found + 1
                index.append(c)
        c = c + 1
    return sampled_bbox


def clip_bbox(src_bbox):
    src_bbox.xmin = max(min(src_bbox.xmin, 1.0), 0.0)
    src_bbox.ymin = max(min(src_bbox.ymin, 1.0), 0.0)
    src_bbox.xmax = max(min(src_bbox.xmax, 1.0), 0.0)
    src_bbox.ymax = max(min(src_bbox.ymax, 1.0), 0.0)
    return src_bbox


def meet_emit_constraint(src_bbox, sample_bbox):
    center_x = (src_bbox.xmax + src_bbox.xmin) / 2
    center_y = (src_bbox.ymax + src_bbox.ymin) / 2
    if center_x >= sample_bbox.xmin and         center_x <= sample_bbox.xmax and         center_y >= sample_bbox.ymin and         center_y <= sample_bbox.ymax:
        return True
    return False


def transform_labels(bbox_labels, sample_bbox):
    proj_bbox = bbox(0, 0, 0, 0)
    sample_labels = []
    for i in range(len(bbox_labels)):
        sample_label = []
        object_bbox = bbox(bbox_labels[i][1], bbox_labels[i][2],
                           bbox_labels[i][3], bbox_labels[i][4])
        if not meet_emit_constraint(object_bbox, sample_bbox):
            continue
        sample_width = sample_bbox.xmax - sample_bbox.xmin
        sample_height = sample_bbox.ymax - sample_bbox.ymin
        proj_bbox.xmin = (object_bbox.xmin - sample_bbox.xmin) / sample_width
        proj_bbox.ymin = (object_bbox.ymin - sample_bbox.ymin) / sample_height
        proj_bbox.xmax = (object_bbox.xmax - sample_bbox.xmin) / sample_width
        proj_bbox.ymax = (object_bbox.ymax - sample_bbox.ymin) / sample_height
        proj_bbox = clip_bbox(proj_bbox)
        if bbox_area(proj_bbox) > 0:
            sample_label.append(bbox_labels[i][0])
            sample_label.append(float(proj_bbox.xmin))
            sample_label.append(float(proj_bbox.ymin))
            sample_label.append(float(proj_bbox.xmax))
            sample_label.append(float(proj_bbox.ymax))
            #sample_label.append(bbox_labels[i][5])
            sample_label = sample_label + bbox_labels[i][5:]
            sample_labels.append(sample_label)
    return sample_labels


def crop_image(img, bbox_labels, sample_bbox, image_width, image_height):
    sample_bbox = clip_bbox(sample_bbox)
    xmin = int(sample_bbox.xmin * image_width)
    xmax = int(sample_bbox.xmax * image_width)
    ymin = int(sample_bbox.ymin * image_height)
    ymax = int(sample_bbox.ymax * image_height)
    sample_img = img[ymin:ymax, xmin:xmax]
    sample_labels = transform_labels(bbox_labels, sample_bbox)
    return sample_img, sample_labels


def random_brightness(img, settings):
    prob = np.random.uniform(0, 1)
    if prob < settings._brightness_prob:
        delta = np.random.uniform(-settings._brightness_delta,
                               settings._brightness_delta) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    return img


def random_contrast(img, settings):
    prob = np.random.uniform(0, 1)
    if prob < settings._contrast_prob:
        delta = np.random.uniform(-settings._contrast_delta,
                               settings._contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    return img


def random_saturation(img, settings):
    prob = np.random.uniform(0, 1)
    if prob < settings._saturation_prob:
        delta = np.random.uniform(-settings._saturation_delta,
                               settings._saturation_delta) + 1
        img = ImageEnhance.Color(img).enhance(delta)
    return img


def random_hue(img, settings):
    prob = np.random.uniform(0, 1)
    if prob < settings._hue_prob:
        delta = np.random.uniform(-settings._hue_delta, settings._hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    return img


def distort_image(img, settings):
    prob = np.random.uniform(0, 1)
    # Apply different distort order
    if prob > 0.5:
        img = random_brightness(img, settings)
        img = random_contrast(img, settings)
        img = random_saturation(img, settings)
        img = random_hue(img, settings)
    else:
        img = random_brightness(img, settings)
        img = random_saturation(img, settings)
        img = random_hue(img, settings)
        img = random_contrast(img, settings)
    return img


def expand_image(img, bbox_labels, img_width, img_height, settings):
    prob = np.random.uniform(0, 1)
    if prob < settings._expand_prob:
        if settings._expand_max_ratio - 1 >= 0.01:
            expand_ratio = np.random.uniform(1, settings._expand_max_ratio)
            height = int(img_height * expand_ratio)
            width = int(img_width * expand_ratio)
            h_off = math.floor(np.random.uniform(0, height - img_height))
            w_off = math.floor(np.random.uniform(0, width - img_width))
            expand_bbox = bbox(-w_off / img_width, -h_off / img_height,
                               (width - w_off) / img_width,
                               (height - h_off) / img_height)
            expand_img = np.ones((height, width, 3))
            expand_img = np.uint8(expand_img * np.squeeze(settings._img_mean))
            expand_img = Image.fromarray(expand_img)
            expand_img.paste(img, (int(w_off), int(h_off)))
            bbox_labels = transform_labels(bbox_labels, expand_bbox)
            return expand_img, bbox_labels, width, height
    return img, bbox_labels, img_width, img_height


# #### 1.3.5 数据读取`reader`
# 通过代码：
# ```python
# json_category_id_to_contiguous_id = {
#     v: i + 1
#     for i, v in enumerate(coco_api.getCatIds())
# }
# ```
# 在读取数据时，将类别从91变为81。

# In[8]:


#coding:utf-8
#定义reader
import xml.etree.ElementTree
import copy
import six
from PIL import Image
from PIL import ImageDraw

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
        if 'pascalvoc' in dataset:
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
            img = distort_image(img, settings)
        if settings._apply_expand:
            img, bbox_labels, img_width, img_height = expand_image(
                img, bbox_labels, img_width, img_height, settings)
        # sampling
        batch_sampler = []
        # hard-code here
        batch_sampler.append(
            sampler(1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0))
        batch_sampler.append(
            sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.1, 0.0))
        batch_sampler.append(
            sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.3, 0.0))
        batch_sampler.append(
            sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.5, 0.0))
        batch_sampler.append(
            sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.7, 0.0))
        batch_sampler.append(
            sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.9, 0.0))
        batch_sampler.append(
            sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.0, 1.0))
        sampled_bbox = generate_batch_samples(batch_sampler,
                                                         bbox_labels)

        img = np.array(img)
        if len(sampled_bbox) > 0:
            idx = int(np.random.uniform(0, len(sampled_bbox)))
            img, sampled_labels = crop_image(
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


def coco(settings, coco_api, file_list, mode, batch_size, shuffle, data_dir):
    from pycocotools.coco import COCO

    def reader():
        if mode == 'train' and shuffle:
            np.random.shuffle(file_list)
        batch_out = []
        for image in file_list:
            image_name = image['file_name']
            image_path = os.path.join(data_dir, image_name)
            if not os.path.exists(image_path):
                raise ValueError("%s is not exist, you should specify "
                                 "data path correctly." % image_path)
            im = Image.open(image_path)
            if im.mode == 'L':
                im = im.convert('RGB')
            im_width, im_height = im.size
            im_id = image['id']

            # layout: category_id | xmin | ymin | xmax | ymax | iscrowd
            bbox_labels = []
            annIds = coco_api.getAnnIds(imgIds=image['id'])
            anns = coco_api.loadAnns(annIds)
            
            json_category_id_to_contiguous_id = {
                v: i + 1
                for i, v in enumerate(coco_api.getCatIds())
            }
            
            for ann in anns:
                bbox_sample = []
                # start from 1, leave 0 to background
                bbox_sample.append(float(json_category_id_to_contiguous_id[ann['category_id']]))
                # bbox_sample.append(float(ann['category_id']))
                bbox = ann['bbox']
                xmin, ymin, w, h = bbox
                xmax = xmin + w
                ymax = ymin + h
                bbox_sample.append(float(xmin) / im_width)
                bbox_sample.append(float(ymin) / im_height)
                bbox_sample.append(float(xmax) / im_width)
                bbox_sample.append(float(ymax) / im_height)
                bbox_sample.append(float(ann['iscrowd']))
                bbox_labels.append(bbox_sample)
            im, sample_labels = preprocess(im, bbox_labels, mode, settings)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) == 0: continue
            im = im.astype('float32')
            boxes = sample_labels[:, 1:5]
            lbls = sample_labels[:, 0].astype('int32')
            iscrowd = sample_labels[:, -1].astype('int32')
            if 'cocoMAP' in settings.ap_version:
                batch_out.append((im, boxes, lbls, iscrowd,
                                  [im_id, im_width, im_height]))
            else:
                batch_out.append((im, boxes, lbls, iscrowd))

            if len(batch_out) == batch_size:
                yield batch_out
                batch_out = []

        if mode == 'test' and len(batch_out) > 1:
            yield batch_out
            batch_out = []

    return reader


def pascalvoc(settings, file_list, mode, batch_size, shuffle):
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
                difficult = float(object.find('difficult').text)
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


def reader_train(settings,
          file_list,
          batch_size,
          shuffle=True,
          use_multiprocess=True,
          num_workers=8,
          enable_ce=False):
    file_path = os.path.join(settings.data_dir, file_list)
    print("reader_train->file_path:{}".format(file_path))
    readers = []
    if 'coco' in settings.dataset:
        # cocoapi
        from pycocotools.coco import COCO
        coco_api = COCO(file_path)
        image_ids = coco_api.getImgIds()
        images = coco_api.loadImgs(image_ids)
        np.random.shuffle(images)
        if '2014' in file_list:
            sub_dir = "train2014"
        elif '2017' in file_list:
            sub_dir = "train2017"
        data_dir = os.path.join(settings.data_dir, sub_dir)

        n = int(math.ceil(len(images) // num_workers)) if use_multiprocess             else len(images)
        image_lists = [images[i:i + n] for i in range(0, len(images), n)]
        for l in image_lists:
            readers.append(
                coco(settings, coco_api, l, 'train', batch_size, shuffle,
                     data_dir))
    else:
        images = [line.strip() for line in open(file_path)]
        np.random.shuffle(images)
        n = int(math.ceil(len(images) // num_workers)) if use_multiprocess             else len(images)
        image_lists = [images[i:i + n] for i in range(0, len(images), n)]
        for l in image_lists:
            readers.append(pascalvoc(settings, l, 'train', batch_size, shuffle))
    print("use_multiprocess ", use_multiprocess)
    if use_multiprocess:
        return paddle.reader.multiprocess_reader(readers, False)
    else:
        return readers[0]


def reader_test(settings, file_list, batch_size):
    file_list = os.path.join(settings.data_dir, file_list)
    if 'coco' in settings.dataset:
        from pycocotools.coco import COCO
        coco_api = COCO(file_list)
        image_ids = coco_api.getImgIds()
        images = coco_api.loadImgs(image_ids)
        if '2014' in file_list:
            sub_dir = "val2014"
        elif '2017' in file_list:
            sub_dir = "val2017"
        data_dir = os.path.join(settings.data_dir, sub_dir)
        return coco(settings, coco_api, images, 'test', batch_size, False,
                    data_dir)
    else:
        image_list = [line.strip() for line in open(file_list)]
        return pascalvoc(settings, image_list, 'test', batch_size, False)


def reader_infer(settings, image_path):
    def reader():
        if not os.path.exists(image_path):
            raise ValueError("%s is not exist, you should specify "
                             "data path correctly." % image_path)
        img = Image.open(image_path)
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


# #### 1.3.6 设置优化器策略`optimizer`

# In[9]:


#coding:utf-8
#设置优化策略
def optimizer_setting(train_params):
    train_batch_size = train_params["batch_size"]
    train_iters = train_params["train_images"] // train_batch_size
    train_lr = train_params["lr"]
    boundaries = [i * train_iters  for i in train_params["lr_epochs"]]
    values = [ i * train_lr for i in train_params["lr_decay"]]

    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        regularization=fluid.regularizer.L2Decay(0.00005), )

    return optimizer


# #### 1.3.7 自定义网络结构`mobilev2 net ssd`（loss降不下去，效果不好）

# In[11]:


# from paddle.fluid.initializer import MSRA
# from paddle.fluid.param_attr import ParamAttr

# class MobileNetV2SSD():
#     def __init__(self, img, num_classes, img_shape, change_depth=False):
#         self.img = img
#         self.num_classes = num_classes
#         self.img_shape = img_shape
#         self.change_depth = change_depth

#     def net(self, scale=1.0):
#         change_depth = self.change_depth
#         # if change_depth is True, the new depth is 1.4 times as deep as before.
#         bottleneck_params_list = [
#             (1, 16, 1, 1),
#             (6, 24, 2, 2),
#             (6, 32, 3, 2),
#             (6, 64, 4, 2),
#             (6, 96, 3, 1),
#             (6, 160, 3, 2),
#             (6, 320, 1, 1),
#         ] if change_depth == False else [
#             (1, 16, 1, 1),
#             (6, 24, 2, 2),
#             (6, 32, 5, 2),
#             (6, 64, 7, 2),
#             (6, 96, 5, 1),
#             (6, 160, 3, 2),
#             (6, 320, 1, 1),
#         ]

#         # conv1
#         input = self.conv_bn_layer(
#             self.img,
#             num_filters=int(32 * scale),
#             filter_size=3,
#             stride=2,
#             padding=1,
#             if_act=True,
#             name='conv1_1')

#         # bottleneck sequences
#         i = 1
#         in_c = int(32 * scale)
#         module11 = None
#         for layer_setting in bottleneck_params_list:
#             t, c, n, s = layer_setting
#             i += 1
#             input = self.invresi_blocks(
#                 input=input,
#                 in_c=in_c,
#                 t=t,
#                 c=int(c * scale),
#                 n=n,
#                 s=s,
#                 name='conv' + str(i))
#             if i==6:
#                 # 19x19
#                 module11 = self.conv_bn_layer(input=input,num_filters=512,filter_size=1,stride=1,padding=0,if_act=True,name='ssd1')
#             in_c = int(c * scale)
#         # last_conv
#         tmp = self.conv_bn_layer(
#             input=input,
#             num_filters=int(1280 * scale) if scale > 1.0 else 1280,
#             filter_size=1,
#             stride=1,
#             padding=0,
#             if_act=True,
#             name='conv9')
#         module13 = self.conv_bn_layer(input=tmp,num_filters=1024,filter_size=1,stride=1,padding=0,if_act=True,name='ssd2')
#         # 10x10
#         module14 = self.extra_block(module13, 256, 512, 1, 2, scale)
#         # 5x5
#         module15 = self.extra_block(module14, 128, 256, 1, 2, scale)
#         # 3x3
#         module16 = self.extra_block(module15, 128, 256, 1, 2, scale)
#         # 2x2
#         module17 = self.extra_block(module16, 64, 128, 1, 2, scale)

#         # mbox_locs：预测的输入框的位置
#         # mbox_confs：预测框对输入的置信度
#         # box：PriorBox输出的先验框
#         # box_var：PriorBox的扩展方差
#         mbox_locs, mbox_confs, box, box_var = fluid.layers.multi_box_head(
#             inputs=[
#                 module11, module13, module14, module15, module16, module17
#             ],
#             image=self.img,
#             num_classes=self.num_classes,
#             min_ratio=20,
#             max_ratio=90,
#             min_sizes=[60.0, 105.0, 150.0, 195.0, 240.0, 285.0],
#             max_sizes=[[], 150.0, 195.0, 240.0, 285.0, 300.0],
#             aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2., 3.],
#                           [2., 3.]],
#             base_size=self.img_shape[2],
#             offset=0.5,
#             flip=True)

#         return mbox_locs, mbox_confs, box, box_var

#     def conv_bn_layer(self,
#                       input,
#                       filter_size,
#                       num_filters,
#                       stride,
#                       padding,
#                       channels=None,
#                       num_groups=1,
#                       if_act=True,
#                       name=None,
#                       use_cudnn=True):
#         conv = fluid.layers.conv2d(
#             input=input,
#             num_filters=num_filters,
#             filter_size=filter_size,
#             stride=stride,
#             padding=padding,
#             groups=num_groups,
#             act=None,
#             use_cudnn=use_cudnn,
#             param_attr=ParamAttr(name=name + '_weights'),
#             bias_attr=False)
#         bn_name = name + '_bn'
#         bn = fluid.layers.batch_norm(
#             input=conv,
#             param_attr=ParamAttr(name=bn_name + "_scale"),
#             bias_attr=ParamAttr(name=bn_name + "_offset"),
#             moving_mean_name=bn_name + '_mean',
#             moving_variance_name=bn_name + '_variance')
#         if if_act:
#             return fluid.layers.relu6(bn)
#         else:
#             return bn

#     def shortcut(self, input, data_residual):
#         return fluid.layers.elementwise_add(input, data_residual)

#     def inverted_residual_unit(self,
#                               input,
#                               num_in_filter,
#                               num_filters,
#                               ifshortcut,
#                               stride,
#                               filter_size,
#                               padding,
#                               expansion_factor,
#                               name=None):
#         num_expfilter = int(round(num_in_filter * expansion_factor))

#         channel_expand = self.conv_bn_layer(
#             input=input,
#             num_filters=num_expfilter,
#             filter_size=1,
#             stride=1,
#             padding=0,
#             num_groups=1,
#             if_act=True,
#             name=name + '_expand')

#         bottleneck_conv = self.conv_bn_layer(
#             input=channel_expand,
#             num_filters=num_expfilter,
#             filter_size=filter_size,
#             stride=stride,
#             padding=padding,
#             num_groups=num_expfilter,
#             if_act=True,
#             name=name + '_dwise',
#             use_cudnn=False)

#         linear_out = self.conv_bn_layer(
#             input=bottleneck_conv,
#             num_filters=num_filters,
#             filter_size=1,
#             stride=1,
#             padding=0,
#             num_groups=1,
#             if_act=False,
#             name=name + '_linear')
#         if ifshortcut:
#             out = self.shortcut(input=input, data_residual=linear_out)
#             return out
#         else:
#             return linear_out

#     def invresi_blocks(self, input, in_c, t, c, n, s, name=None):
#         first_block = self.inverted_residual_unit(
#             input=input,
#             num_in_filter=in_c,
#             num_filters=c,
#             ifshortcut=False,
#             stride=s,
#             filter_size=3,
#             padding=1,
#             expansion_factor=t,
#             name=name + '_1')

#         last_residual_block = first_block
#         last_c = c

#         for i in range(1, n):
#             last_residual_block = self.inverted_residual_unit(
#                 input=last_residual_block,
#                 num_in_filter=last_c,
#                 num_filters=c,
#                 ifshortcut=True,
#                 stride=1,
#                 filter_size=3,
#                 padding=1,
#                 expansion_factor=t,
#                 name=name + '_' + str(i + 1))
#         return last_residual_block

#     def extra_block(self, input, num_filters1, num_filters2, num_groups, stride,
#                     scale):
#         # 1x1 conv
#         pointwise_conv = self.conv_bn(
#             input=input,
#             filter_size=1,
#             num_filters=int(num_filters1 * scale),
#             stride=1,
#             num_groups=int(num_groups * scale),
#             padding=0)

#         # 3x3 conv
#         normal_conv = self.conv_bn(
#             input=pointwise_conv,
#             filter_size=3,
#             num_filters=int(num_filters2 * scale),
#             stride=2,
#             num_groups=int(num_groups * scale),
#             padding=1)
#         return normal_conv

#     def conv_bn(self,
#                 input,
#                 filter_size,
#                 num_filters,
#                 stride,
#                 padding,
#                 channels=None,
#                 num_groups=1,
#                 act='relu',
#                 use_cudnn=True):
#         parameter_attr = ParamAttr(learning_rate=0.1, initializer=MSRA())
#         conv = fluid.layers.conv2d(
#             input=input,
#             num_filters=num_filters,
#             filter_size=filter_size,
#             stride=stride,
#             padding=padding,
#             groups=num_groups,
#             act=None,
#             use_cudnn=use_cudnn,
#             param_attr=parameter_attr,
#             bias_attr=False)
#         return fluid.layers.batch_norm(input=conv, act=act)

# def MobileNetV2_x0_25(img, num_classes, img_shape):
#     model = MobileNetV2SSD(img, num_classes, img_shape)
#     return model.net(scale=0.25)


# def MobileNetV2_x0_5(img, num_classes, img_shape):
#     model = MobileNetV2SSD(img, num_classes, img_shape)
#     return model.net(scale=0.5)


# def MobileNetV2_x1_0(img, num_classes, img_shape):
#     model = MobileNetV2SSD(img, num_classes, img_shape)
#     return model.net(scale=1.0)


# def MobileNetV2_x1_5(img, num_classes, img_shape):
#     model = MobileNetV2SSD(img, num_classes, img_shape)
#     return model.net(scale=1.5)


# def MobileNetV2_x2_0(img, num_classes, img_shape):
#     model = MobileNetV2SSD(img, num_classes, img_shape)
#     return model.net(scale=2.0)


# def MobileNetV2_scale(img, num_classes, img_shape):
#     model = MobileNetV2SSD(img, num_classes, img_shape, change_depth=True)
#     return model.net(scale=1.2)


# #### 1.3.8 定义网络结构`mobile net ssd`

# In[12]:


import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr


class MobileNetSSD:
    def __init__(self, img, num_classes, img_shape):
        self.img = img
        self.num_classes = num_classes
        self.img_shape = img_shape

    def ssd_net(self, scale=1.0):
        # 300x300
        tmp = self.conv_bn(self.img, 3, int(32 * scale), 2, 1, 3)
        # 150x150
        tmp = self.depthwise_separable(tmp, 32, 64, 32, 1, scale)
        tmp = self.depthwise_separable(tmp, 64, 128, 64, 2, scale)
        # 75x75
        tmp = self.depthwise_separable(tmp, 128, 128, 128, 1, scale)
        tmp = self.depthwise_separable(tmp, 128, 256, 128, 2, scale)
        # 38x38
        tmp = self.depthwise_separable(tmp, 256, 256, 256, 1, scale)
        tmp = self.depthwise_separable(tmp, 256, 512, 256, 2, scale)

        # 19x19
        for i in range(5):
            tmp = self.depthwise_separable(tmp, 512, 512, 512, 1, scale)
        module11 = tmp
        tmp = self.depthwise_separable(tmp, 512, 1024, 512, 2, scale)

        # 10x10
        module13 = self.depthwise_separable(tmp, 1024, 1024, 1024, 1, scale)
        module14 = self.extra_block(module13, 256, 512, 1, 2, scale)
        # 5x5
        module15 = self.extra_block(module14, 128, 256, 1, 2, scale)
        # 3x3
        module16 = self.extra_block(module15, 128, 256, 1, 2, scale)
        # 2x2
        module17 = self.extra_block(module16, 64, 128, 1, 2, scale)

        mbox_locs, mbox_confs, box, box_var = fluid.layers.multi_box_head(
            inputs=[
                module11, module13, module14, module15, module16, module17
            ],
            image=self.img,
            num_classes=self.num_classes,
            min_ratio=20,
            max_ratio=90,
            min_sizes=[60.0, 105.0, 150.0, 195.0, 240.0, 285.0],
            max_sizes=[[], 150.0, 195.0, 240.0, 285.0, 300.0],
            aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2., 3.],
                          [2., 3.]],
            base_size=self.img_shape[2],
            offset=0.5,
            flip=True)

        return mbox_locs, mbox_confs, box, box_var

    def conv_bn(self,
                input,
                filter_size,
                num_filters,
                stride,
                padding,
                channels=None,
                num_groups=1,
                act='relu',
                use_cudnn=True):
        parameter_attr = ParamAttr(learning_rate=0.1, initializer=MSRA())
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=parameter_attr,
            bias_attr=False)
        return fluid.layers.batch_norm(input=conv, act=act)

    def depthwise_separable(self, input, num_filters1, num_filters2, num_groups,
                            stride, scale):
        depthwise_conv = self.conv_bn(
            input=input,
            filter_size=3,
            num_filters=int(num_filters1 * scale),
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            use_cudnn=False)

        pointwise_conv = self.conv_bn(
            input=depthwise_conv,
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0)
        return pointwise_conv

    def extra_block(self, input, num_filters1, num_filters2, num_groups, stride,
                    scale):
        # 1x1 conv
        pointwise_conv = self.conv_bn(
            input=input,
            filter_size=1,
            num_filters=int(num_filters1 * scale),
            stride=1,
            num_groups=int(num_groups * scale),
            padding=0)

        # 3x3 conv
        normal_conv = self.conv_bn(
            input=pointwise_conv,
            filter_size=3,
            num_filters=int(num_filters2 * scale),
            stride=2,
            num_groups=int(num_groups * scale),
            padding=1)
        return normal_conv


def build_mobilenet_ssd(img, num_classes, img_shape):
    ssd_model = MobileNetSSD(img, num_classes, img_shape)
    return ssd_model.ssd_net()


# #### 1.3.9 修改ssd_loss函数，替换原有框架中ssd_loss函数
# 主要拆分其中nn.softmax_with_cross_entropy函数为nn.softmax和nn.cross_entropy两个函数，避免后续量化、剪枝操作后的训练中，loss为NAN的错误。

# In[13]:


from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers import nn, iou_similarity, bipartite_match, target_assign, tensor, box_coder


def ssd_loss(location,
             confidence,
             gt_box,
             gt_label,
             prior_box,
             prior_box_var=None,
             background_label=0,
             overlap_threshold=0.5,
             neg_pos_ratio=3.0,
             neg_overlap=0.5,
             loc_loss_weight=1.0,
             conf_loss_weight=1.0,
             match_type='per_prediction',
             mining_type='max_negative',
             normalize=True,
             sample_size=None):

    helper = LayerHelper('ssd_loss', **locals())
    if mining_type != 'max_negative':
        raise ValueError("Only support mining_type == max_negative now.")

    num, num_prior, num_class = confidence.shape
    conf_shape = nn.shape(confidence)

    def __reshape_to_2d(var):
        return nn.flatten(x=var, axis=2)

    # 1. Find matched boundding box by prior box.
    #   1.1 Compute IOU similarity between ground-truth boxes and prior boxes.
    iou = iou_similarity(x=gt_box, y=prior_box)
    #   1.2 Compute matched boundding box by bipartite matching algorithm.
    matched_indices, matched_dist = bipartite_match(iou, match_type,
                                                    overlap_threshold)

    # 2. Compute confidence for mining hard examples
    # 2.1. Get the target label based on matched indices
    gt_label = nn.reshape(
        x=gt_label, shape=(len(gt_label.shape) - 1) * (0, ) + (-1, 1))
    gt_label.stop_gradient = True
    target_label, _ = target_assign(
        gt_label, matched_indices, mismatch_value=background_label)
    # 2.2. Compute confidence loss.
    # Reshape confidence to 2D tensor.
    confidence = __reshape_to_2d(confidence)
    target_label = tensor.cast(x=target_label, dtype='int64')
    target_label = __reshape_to_2d(target_label)
    target_label.stop_gradient = True
    #conf_loss = nn.softmax_with_cross_entropy(confidence, target_label)
    conf_softmax = nn.softmax(confidence,use_cudnn=False)
    conf_loss = nn.cross_entropy(conf_softmax, target_label)
    # 3. Mining hard examples
    actual_shape = nn.slice(conf_shape, axes=[0], starts=[0], ends=[2])
    actual_shape.stop_gradient = True
    conf_loss = nn.reshape(
        x=conf_loss, shape=(num, num_prior), actual_shape=actual_shape)
    conf_loss.stop_gradient = True
    neg_indices = helper.create_variable_for_type_inference(dtype='int32')
    dtype = matched_indices.dtype
    updated_matched_indices = helper.create_variable_for_type_inference(
        dtype=dtype)
    helper.append_op(
        type='mine_hard_examples',
        inputs={
            'ClsLoss': conf_loss,
            'LocLoss': None,
            'MatchIndices': matched_indices,
            'MatchDist': matched_dist,
        },
        outputs={
            'NegIndices': neg_indices,
            'UpdatedMatchIndices': updated_matched_indices
        },
        attrs={
            'neg_pos_ratio': neg_pos_ratio,
            'neg_dist_threshold': neg_overlap,
            'mining_type': mining_type,
            'sample_size': sample_size,
        })

    # 4. Assign classification and regression targets
    # 4.1. Encoded bbox according to the prior boxes.
    encoded_bbox = box_coder(
        prior_box=prior_box,
        prior_box_var=prior_box_var,
        target_box=gt_box,
        code_type='encode_center_size')
    # 4.2. Assign regression targets
    target_bbox, target_loc_weight = target_assign(
        encoded_bbox, updated_matched_indices, mismatch_value=background_label)
    # 4.3. Assign classification targets
    target_label, target_conf_weight = target_assign(
        gt_label,
        updated_matched_indices,
        negative_indices=neg_indices,
        mismatch_value=background_label)

    # 5. Compute loss.
    # 5.1 Compute confidence loss.
    target_label = __reshape_to_2d(target_label)
    target_label = tensor.cast(x=target_label, dtype='int64')

    # conf_loss = nn.softmax_with_cross_entropy(confidence, target_label)
    conf_softmax = nn.softmax(confidence, use_cudnn=False)
    conf_loss = nn.cross_entropy(conf_softmax, target_label)
    target_conf_weight = __reshape_to_2d(target_conf_weight)
    conf_loss = conf_loss * target_conf_weight

    # the target_label and target_conf_weight do not have gradient.
    target_label.stop_gradient = True
    target_conf_weight.stop_gradient = True

    # 5.2 Compute regression loss.
    location = __reshape_to_2d(location)
    target_bbox = __reshape_to_2d(target_bbox)

    loc_loss = nn.smooth_l1(location, target_bbox)
    target_loc_weight = __reshape_to_2d(target_loc_weight)
    loc_loss = loc_loss * target_loc_weight

    # the target_bbox and target_loc_weight do not have gradient.
    target_bbox.stop_gradient = True
    target_loc_weight.stop_gradient = True

    # 5.3 Compute overall weighted loss.
    loss = conf_loss_weight * conf_loss + loc_loss_weight * loc_loss
    # reshape to [N, Np], N is the batch size and Np is the prior box number.
    loss = nn.reshape(x=loss, shape=(num, num_prior), actual_shape=actual_shape)
    loss = nn.reduce_sum(loss, dim=1, keep_dim=True)
    if normalize:
        normalizer = nn.reduce_sum(target_loc_weight)
        loss = loss / normalizer

    return loss


# #### 1.3.10 构建训练用到的`program`

# In[14]:


#coding:utf-8
#建立训练(预测)主程序,从mobilenet_ssd中读取预训练模型参数
def build_program(main_prog, startup_prog, train_params, is_train):
    train_image_shape = train_params['image_shape']
    train_class_num = train_params['class_num']
    train_ap_version = train_params['ap_version']
    outs = []
    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=64,
            shapes=[[-1] + train_image_shape, [-1, 4], [-1, 1], [-1, 1]],
            lod_levels=[0, 1, 1, 1],
            dtypes=["float32", "float32", "int32", "int32"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, gt_box, gt_label, difficult = fluid.layers.read_file(py_reader)
            locs, confs, box, box_var = build_mobilenet_ssd(image, train_class_num, train_image_shape)
            # locs, confs, box, box_var = MobileNetV2_x1_0(image, train_class_num, train_image_shape)
            if is_train:
                with fluid.unique_name.guard("train"):
                    loss = ssd_loss(locs, confs, gt_box, gt_label, box,
                        box_var)
                    loss = fluid.layers.reduce_sum(loss)
                    optimizer = optimizer_setting(train_params)
                    optimizer.minimize(loss)
                outs = [py_reader, loss]
            else:
                with fluid.unique_name.guard("inference"):
                    nmsed_out = fluid.layers.detection_output(
                        locs, confs, box, box_var, nms_threshold=0.45)
                    map_eval = fluid.metrics.DetectionMAP(
                        nmsed_out,
                        gt_label,
                        gt_box,
                        difficult,
                        train_class_num,
                        overlap_threshold=0.5,
                        evaluate_difficult=False,
                        ap_version=train_ap_version)
                # nmsed_out and image is used to save mode for inference
                outs = [py_reader, map_eval, nmsed_out, image]
    return outs


# #### 1.3.11 构建训练逻辑代码

# In[15]:


#coding:utf-8
#构建训练逻辑代码
def start_train(data_args,
          train_params,
          train_file_list,
          val_file_list):

    train_model_save_dir = g_model_save_dir
    train_pretrained_model = g_pretrained_model
    train_use_gpu = g_use_gpu
    train_parallel = g_parallel

    is_shuffle = True

    if not train_use_gpu:
        devices_num = int(os.environ.get('CPU_NUM',
                          multiprocessing.cpu_count()))
    else:
        devices_num = fluid.core.get_cuda_device_count()

    batch_size = train_params['batch_size']
    epoc_num = train_params['epoc_num']
    batch_size_per_device = batch_size // devices_num
    num_workers = 8

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()

    #建立训练、测试数据提供器，获取loss
    train_py_reader, loss = build_program(
        main_prog=train_prog,
        startup_prog=startup_prog,
        train_params=train_params,
        is_train=True)
    test_py_reader, map_eval, _, _ = build_program(
        main_prog=test_prog,
        startup_prog=startup_prog,
        train_params=train_params,
        is_train=False)
    test_prog = test_prog.clone(for_test=True)
    
    place = fluid.CUDAPlace(0) if train_use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    
    exe.run(startup_prog)

    if train_pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(train_pretrained_model, var.name))
        fluid.io.load_vars(exe, train_pretrained_model, main_program=train_prog,
                           predicate=if_exist)

    if train_parallel:
        loss.persistable = True
        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = True
        build_strategy.memory_optimize = True
        train_exe = fluid.ParallelExecutor(main_program=train_prog,
            use_cuda=train_use_gpu, loss_name=loss.name, build_strategy=build_strategy)
    
    
    test_reader = reader_test(data_args, val_file_list, batch_size)
    test_py_reader.decorate_paddle_reader(test_reader)


    def save_model(postfix, main_prog):
        model_path = os.path.join(train_model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        print('save models to %s' % (model_path))
        fluid.io.save_persistables(exe, model_path, main_program=main_prog)

    best_map = 0.
    test_map = None
    def test(epoc_id, best_map):
        _, accum_map = map_eval.get_map_var()
        map_eval.reset(exe)
        every_epoc_map=[] # for CE
        test_py_reader.start()
        try:
            batch_id = 0
            while True:
                test_map, = exe.run(test_prog, fetch_list=[accum_map])
                if batch_id % 10 == 0:
                    every_epoc_map.append(test_map)
                    print("Batch {0}, map {1}".format(batch_id, test_map))
                batch_id += 1
        except fluid.core.EOFException:
            test_py_reader.reset()
            
        mean_map = np.mean(every_epoc_map)
        print("Epoc {0}, test map {1}".format(epoc_id, test_map[0]))
        if test_map[0] > best_map:
            best_map = test_map[0]
            save_model('best_model', test_prog)
        return best_map, mean_map


    total_time = 0.0
    for epoc_id in range(epoc_num):
        train_reader = reader_train(data_args,
                                train_file_list,
                                batch_size_per_device,
                                shuffle=is_shuffle,
                                use_multiprocess=g_use_multiprocess,
                                num_workers=num_workers,
                                enable_ce=False)
        train_py_reader.decorate_paddle_reader(train_reader)
        epoch_idx = epoc_id + 1
        start_time = time.time()
        prev_start_time = start_time
        every_epoc_loss = []
        batch_id = 0
        train_py_reader.start()
        while True:
            try:
                prev_start_time = start_time
                start_time = time.time()
                if train_parallel:
                    loss_v, = train_exe.run(fetch_list=[loss.name],return_numpy=False)
                else:
                    loss_v, = exe.run(train_prog, fetch_list=[loss],return_numpy=False)
                loss_v = np.mean(np.array(loss_v))
                every_epoc_loss.append(loss_v)
                if batch_id % 10 == 0:
                    print("Epoc {:d}, batch {:d}, loss {:.6f}, time {:.5f}".format(
                        epoc_id, batch_id, loss_v, start_time - prev_start_time))
                batch_id += 1
            except (fluid.core.EOFException, StopIteration):
                train_reader().close()
                train_py_reader.reset()
                break

        end_time = time.time()
        total_time += end_time - start_time
        if epoc_id % 10 == 0 or epoc_id == epoc_num - 1:
            best_map, mean_map = test(epoc_id, best_map)
            print("Best test map {0}".format(best_map))
            # save model
            save_model(str(epoc_id), train_prog)


# #### 1.3.12 开始训练

# In[16]:


#coding:utf-8
#主程序函数
train_data_dir = g_data_dir
train_dataset = g_dataset
assert train_dataset in ['pascalvoc', 'coco2014', 'coco2017']

if train_dataset == 'coco2017':
    train_file_list = '/home/aistudio/work/coco2017/annotations/instances_train2017.json'
    val_file_list = '/home/aistudio/work/coco2017/annotations/instances_val2017.json'
# if train_dataset == 'coco2017':
#     train_file_list = 'annotations/instances_train2017.json'
#     val_file_list = 'annotations/instances_val2017.json'

mean_BGR_value = [float(m) for m in g_mean_BGR.split(",")]
image_shape_value = [int(m) for m in g_image_shape.split(",")]
train_parameters[train_dataset]['image_shape'] = image_shape_value
train_parameters[train_dataset]['batch_size'] = g_batch_size
train_parameters[train_dataset]['lr'] = g_learning_rate
train_parameters[train_dataset]['epoc_num'] = g_epoc_num
train_parameters[train_dataset]['ap_version'] = g_ap_version

data_args = Settings(
    dataset=g_dataset,
    data_dir=train_data_dir,
    label_file=None,
    resize_h=image_shape_value[1],
    resize_w=image_shape_value[2],
    mean_value=mean_BGR_value,
    apply_distort=True,
    apply_expand=True,
    ap_version = g_ap_version)
start_train(data_args,
      train_parameters[train_dataset],
      train_file_list=train_file_list,
      val_file_list=val_file_list)


# ### 1.4 保存预测模型
# 保存infer模型时，构建fetch_list格式，保证模型的输出符合大赛要求。
# 

# In[16]:


#保存模型
#1.3步骤训练的最佳模型路径
# g_infer_best_dir = '/home/aistudio/work/models/mobilenet_ssd.model/best_model/'
g_infer_best_dir = '/home/aistudio/work/models/mobilenet_ssd.model/34_best/'
#infer模型保存地址
g_infer_model_save_dir = '/home/aistudio/work/models/infer_models/best_models'

def save_model():
    image_shape = [3, 300, 300]
    num_classes = 81
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    locs, confs, box, box_var = build_mobilenet_ssd(image, num_classes,
                                                    image_shape)
    # locs, confs, box, box_var = MobileNetV2_x1_0(image, num_classes,
                                                    # image_shape)
    boxes = fluid.layers.box_coder(
        prior_box=box,
        prior_box_var=box_var,
        target_box=locs,
        code_type='decode_center_size')
    scores = fluid.layers.nn.softmax(input=confs)
    scores = fluid.layers.nn.transpose(scores, perm=[0, 2, 1])
    scores.stop_gradient = True
    
    fetch_list = [boxes] + [scores]

    place = fluid.CUDAPlace(0) if g_use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if g_infer_best_dir:
        def if_exist(var):
            return os.path.exists(os.path.join(g_infer_best_dir, var.name))
        fluid.io.load_vars(exe, g_infer_best_dir, predicate=if_exist)
    # yapf: enable

    # save infer model
    def save_infer_model(image):
        fluid.io.save_inference_model(g_infer_model_save_dir + "_submit_20190914_best", [image.name], fetch_list,exe)

    save_infer_model(image)

save_model()


# ### 1.5 计算模型分数
# 构建测试模型分数脚本，执行脚本，测试分数指标
# 
# 脚本`testssd.sh`内容：
# ```bash
# #!/bin/bash
# python /home/aistudio/work/astar2019/score.py --model_dir /home/aistudio/work/models/pruned_models/20190915_best_1 --data_dir work/coco2017
# ```

# In[1]:


#计算分数
# !cd /home/aistudio/work && tar -xzvf astar2019.tar.gz
get_ipython().system('/bin/bash /home/aistudio/work/astar2019/testssd.sh')


# ## 二、量化、剪枝操作
# ### 2.1 剪枝、量化操作
# 加载前一步骤训练保存的infer模型，进行剪枝、量化操作

# #### 2.1.1 图像预处理工具`image_util`

# In[1]:


#image_util
from PIL import Image, ImageEnhance, ImageDraw
from PIL import ImageFile
import numpy as np
import random
import math

ImageFile.LOAD_TRUNCATED_IMAGES = True  #otherwise IOError raised image file is truncated


class sampler():
    def __init__(self, max_sample, max_trial, min_scale, max_scale,
                 min_aspect_ratio, max_aspect_ratio, min_jaccard_overlap,
                 max_jaccard_overlap):
        self.max_sample = max_sample
        self.max_trial = max_trial
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_jaccard_overlap = min_jaccard_overlap
        self.max_jaccard_overlap = max_jaccard_overlap


class bbox():
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def bbox_area(src_bbox):
    width = src_bbox.xmax - src_bbox.xmin
    height = src_bbox.ymax - src_bbox.ymin
    return width * height


def generate_sample(sampler):
    scale = np.random.uniform(sampler.min_scale, sampler.max_scale)
    aspect_ratio = np.random.uniform(sampler.min_aspect_ratio,
                                  sampler.max_aspect_ratio)
    aspect_ratio = max(aspect_ratio, (scale**2.0))
    aspect_ratio = min(aspect_ratio, 1 / (scale**2.0))

    bbox_width = scale * (aspect_ratio**0.5)
    bbox_height = scale / (aspect_ratio**0.5)
    xmin_bound = 1 - bbox_width
    ymin_bound = 1 - bbox_height
    xmin = np.random.uniform(0, xmin_bound)
    ymin = np.random.uniform(0, ymin_bound)
    xmax = xmin + bbox_width
    ymax = ymin + bbox_height
    sampled_bbox = bbox(xmin, ymin, xmax, ymax)
    return sampled_bbox


def jaccard_overlap(sample_bbox, object_bbox):
    if sample_bbox.xmin >= object_bbox.xmax or             sample_bbox.xmax <= object_bbox.xmin or             sample_bbox.ymin >= object_bbox.ymax or             sample_bbox.ymax <= object_bbox.ymin:
        return 0
    intersect_xmin = max(sample_bbox.xmin, object_bbox.xmin)
    intersect_ymin = max(sample_bbox.ymin, object_bbox.ymin)
    intersect_xmax = min(sample_bbox.xmax, object_bbox.xmax)
    intersect_ymax = min(sample_bbox.ymax, object_bbox.ymax)
    intersect_size = (intersect_xmax - intersect_xmin) * (
        intersect_ymax - intersect_ymin)
    sample_bbox_size = bbox_area(sample_bbox)
    object_bbox_size = bbox_area(object_bbox)
    overlap = intersect_size / (
        sample_bbox_size + object_bbox_size - intersect_size)
    return overlap


def satisfy_sample_constraint(sampler, sample_bbox, bbox_labels):
    if sampler.min_jaccard_overlap == 0 and sampler.max_jaccard_overlap == 0:
        return True
    for i in range(len(bbox_labels)):
        object_bbox = bbox(bbox_labels[i][1], bbox_labels[i][2],
                           bbox_labels[i][3], bbox_labels[i][4])
        overlap = jaccard_overlap(sample_bbox, object_bbox)
        if sampler.min_jaccard_overlap != 0 and                 overlap < sampler.min_jaccard_overlap:
            continue
        if sampler.max_jaccard_overlap != 0 and                 overlap > sampler.max_jaccard_overlap:
            continue
        return True
    return False


def generate_batch_samples(batch_sampler, bbox_labels):
    sampled_bbox = []
    index = []
    c = 0
    for sampler in batch_sampler:
        found = 0
        for i in range(sampler.max_trial):
            if found >= sampler.max_sample:
                break
            sample_bbox = generate_sample(sampler)
            if satisfy_sample_constraint(sampler, sample_bbox, bbox_labels):
                sampled_bbox.append(sample_bbox)
                found = found + 1
                index.append(c)
        c = c + 1
    return sampled_bbox


def clip_bbox(src_bbox):
    src_bbox.xmin = max(min(src_bbox.xmin, 1.0), 0.0)
    src_bbox.ymin = max(min(src_bbox.ymin, 1.0), 0.0)
    src_bbox.xmax = max(min(src_bbox.xmax, 1.0), 0.0)
    src_bbox.ymax = max(min(src_bbox.ymax, 1.0), 0.0)
    return src_bbox


def meet_emit_constraint(src_bbox, sample_bbox):
    center_x = (src_bbox.xmax + src_bbox.xmin) / 2
    center_y = (src_bbox.ymax + src_bbox.ymin) / 2
    if center_x >= sample_bbox.xmin and         center_x <= sample_bbox.xmax and         center_y >= sample_bbox.ymin and         center_y <= sample_bbox.ymax:
        return True
    return False


def transform_labels(bbox_labels, sample_bbox):
    proj_bbox = bbox(0, 0, 0, 0)
    sample_labels = []
    for i in range(len(bbox_labels)):
        sample_label = []
        object_bbox = bbox(bbox_labels[i][1], bbox_labels[i][2],
                           bbox_labels[i][3], bbox_labels[i][4])
        if not meet_emit_constraint(object_bbox, sample_bbox):
            continue
        sample_width = sample_bbox.xmax - sample_bbox.xmin
        sample_height = sample_bbox.ymax - sample_bbox.ymin
        proj_bbox.xmin = (object_bbox.xmin - sample_bbox.xmin) / sample_width
        proj_bbox.ymin = (object_bbox.ymin - sample_bbox.ymin) / sample_height
        proj_bbox.xmax = (object_bbox.xmax - sample_bbox.xmin) / sample_width
        proj_bbox.ymax = (object_bbox.ymax - sample_bbox.ymin) / sample_height
        proj_bbox = clip_bbox(proj_bbox)
        if bbox_area(proj_bbox) > 0:
            sample_label.append(bbox_labels[i][0])
            sample_label.append(float(proj_bbox.xmin))
            sample_label.append(float(proj_bbox.ymin))
            sample_label.append(float(proj_bbox.xmax))
            sample_label.append(float(proj_bbox.ymax))
            #sample_label.append(bbox_labels[i][5])
            sample_label = sample_label + bbox_labels[i][5:]
            sample_labels.append(sample_label)
    return sample_labels


def crop_image(img, bbox_labels, sample_bbox, image_width, image_height):
    sample_bbox = clip_bbox(sample_bbox)
    xmin = int(sample_bbox.xmin * image_width)
    xmax = int(sample_bbox.xmax * image_width)
    ymin = int(sample_bbox.ymin * image_height)
    ymax = int(sample_bbox.ymax * image_height)
    sample_img = img[ymin:ymax, xmin:xmax]
    sample_labels = transform_labels(bbox_labels, sample_bbox)
    return sample_img, sample_labels


def random_brightness(img, settings):
    prob = np.random.uniform(0, 1)
    if prob < settings._brightness_prob:
        delta = np.random.uniform(-settings._brightness_delta,
                               settings._brightness_delta) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    return img


def random_contrast(img, settings):
    prob = np.random.uniform(0, 1)
    if prob < settings._contrast_prob:
        delta = np.random.uniform(-settings._contrast_delta,
                               settings._contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    return img


def random_saturation(img, settings):
    prob = np.random.uniform(0, 1)
    if prob < settings._saturation_prob:
        delta = np.random.uniform(-settings._saturation_delta,
                               settings._saturation_delta) + 1
        img = ImageEnhance.Color(img).enhance(delta)
    return img


def random_hue(img, settings):
    prob = np.random.uniform(0, 1)
    if prob < settings._hue_prob:
        delta = np.random.uniform(-settings._hue_delta, settings._hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    return img


def distort_image(img, settings):
    prob = np.random.uniform(0, 1)
    # Apply different distort order
    if prob > 0.5:
        img = random_brightness(img, settings)
        img = random_contrast(img, settings)
        img = random_saturation(img, settings)
        img = random_hue(img, settings)
    else:
        img = random_brightness(img, settings)
        img = random_saturation(img, settings)
        img = random_hue(img, settings)
        img = random_contrast(img, settings)
    return img


def expand_image(img, bbox_labels, img_width, img_height, settings):
    prob = np.random.uniform(0, 1)
    if prob < settings._expand_prob:
        if settings._expand_max_ratio - 1 >= 0.01:
            expand_ratio = np.random.uniform(1, settings._expand_max_ratio)
            height = int(img_height * expand_ratio)
            width = int(img_width * expand_ratio)
            h_off = math.floor(np.random.uniform(0, height - img_height))
            w_off = math.floor(np.random.uniform(0, width - img_width))
            expand_bbox = bbox(-w_off / img_width, -h_off / img_height,
                               (width - w_off) / img_width,
                               (height - h_off) / img_height)
            expand_img = np.ones((height, width, 3))
            expand_img = np.uint8(expand_img * np.squeeze(settings._img_mean))
            expand_img = Image.fromarray(expand_img)
            expand_img.paste(img, (int(w_off), int(h_off)))
            bbox_labels = transform_labels(bbox_labels, expand_bbox)
            return expand_img, bbox_labels, width, height
    return img, bbox_labels, img_width, img_height

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
        if 'pascalvoc' in dataset:
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


# #### 2.1.2 设置全局变量参数

# In[32]:


args_image_shape = '3,300,300'
args_mean_BGR = '127.5,127.5,127.5'
args_data_dir = '/home/aistudio/work/coco2017'
args_dataset = 'coco2017'
#预剪枝、量化的模型路径
args_pretrained_model = '/home/aistudio/work/models/infer_models/best_models_submit_20190914_best/'
# args_pretrained_model = '/home/aistudio/work/models_20190824/infer_models/best_models_submit_20190822/'
args_ap_version = '11point'


data_dir = args_data_dir
dataset = args_dataset
assert dataset in ['pascalvoc', 'coco2014', 'coco2017']

# for pascalvoc
label_file = 'label_list'
train_file_list = 'trainval.txt'
val_file_list = 'test.txt'

if dataset == 'coco2014':
    train_file_list = 'annotations/instances_train2014.json'
    val_file_list = 'annotations/instances_val2014.json'
elif dataset == 'coco2017':
    train_file_list = 'annotations/instances_train2017.json'
    val_file_list = 'annotations/instances_val2017.json'

mean_BGR = [float(m) for m in args_mean_BGR.split(",")]
image_shape = [int(m) for m in args_image_shape.split(",")]

data_args = Settings(
dataset=args_dataset,
data_dir=data_dir,
label_file=label_file,
resize_h=image_shape[1],
resize_w=image_shape[2],
mean_value=mean_BGR,
apply_distort=True,
apply_expand=True,
ap_version = args_ap_version)


# #### 2.1.3 数据读取`reader`

# In[33]:


import xml.etree.ElementTree
import os
import time
import copy
import six
import math
import numpy as np
from PIL import Image
from PIL import ImageDraw
import paddle





def preprocess(img, bbox_labels, mode, settings):
    img_width, img_height = img.size
    sampled_labels = bbox_labels
    if mode == 'train':
        if settings._apply_distort:
            img = distort_image(img, settings)
        if settings._apply_expand:
            img, bbox_labels, img_width, img_height = expand_image(
                img, bbox_labels, img_width, img_height, settings)
        # sampling
        batch_sampler = []
        # hard-code here
        batch_sampler.append(
            sampler(1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0))
        batch_sampler.append(
            sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.1, 0.0))
        batch_sampler.append(
            sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.3, 0.0))
        batch_sampler.append(
            sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.5, 0.0))
        batch_sampler.append(
            sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.7, 0.0))
        batch_sampler.append(
            sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.9, 0.0))
        batch_sampler.append(
            sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.0, 1.0))
        sampled_bbox = generate_batch_samples(batch_sampler,
                                                         bbox_labels)

        img = np.array(img)
        if len(sampled_bbox) > 0:
            idx = int(np.random.uniform(0, len(sampled_bbox)))
            img, sampled_labels = crop_image(
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


def coco(settings, coco_api, file_list, mode, batch_size, shuffle, data_dir):
    from pycocotools.coco import COCO

    json_category_id_to_contiguous_id = {
        v: i + 1
        for i, v in enumerate(coco_api.getCatIds())
    }
    contiguous_category_id_to_json_id = {
        v: k
        for k, v in json_category_id_to_contiguous_id.items()
    }
    def reader():
        if mode == 'train' and shuffle:
            np.random.shuffle(file_list)
        batch_out = []
        for image in file_list:
            image_name = image['file_name']
            image_path = os.path.join(data_dir, image_name)
            if not os.path.exists(image_path):
                raise ValueError("%s is not exist, you should specify "
                                 "data path correctly." % image_path)
            im = Image.open(image_path)
            if im.mode == 'L':
                im = im.convert('RGB')
            im_width, im_height = im.size
            im_id = image['id']

            # layout: category_id | xmin | ymin | xmax | ymax | iscrowd
            bbox_labels = []
            annIds = coco_api.getAnnIds(imgIds=image['id'])
            anns = coco_api.loadAnns(annIds)
            for ann in anns:
                bbox_sample = []
                # start from 1, leave 0 to background
                # bbox_sample.append(float(ann['category_id']))
                bbox_sample.append(float(json_category_id_to_contiguous_id[ann['category_id']]))
                bbox = ann['bbox']
                xmin, ymin, w, h = bbox
                xmax = xmin + w
                ymax = ymin + h
                bbox_sample.append(float(xmin) / im_width)
                bbox_sample.append(float(ymin) / im_height)
                bbox_sample.append(float(xmax) / im_width)
                bbox_sample.append(float(ymax) / im_height)
                bbox_sample.append(float(ann['iscrowd']))
                bbox_labels.append(bbox_sample)
            im, sample_labels = preprocess(im, bbox_labels, mode, settings)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) == 0: continue
            im = im.astype('float32')
            boxes = sample_labels[:, 1:5]
            lbls = sample_labels[:, 0].astype('int32')
            iscrowd = sample_labels[:, -1].astype('int32')
            if 'cocoMAP' in settings.ap_version:
                batch_out.append((im, boxes, lbls, iscrowd,
                                  [im_id, im_width, im_height]))
            else:
                batch_out.append((im, boxes, lbls, iscrowd))

            if len(batch_out) == batch_size:
                yield batch_out
                batch_out = []

        if mode == 'test' and len(batch_out) > 1:
            yield batch_out
            batch_out = []

    return reader


def pascalvoc(settings, file_list, mode, batch_size, shuffle):
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
                difficult = float(object.find('difficult').text)
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


def train_data_reader(settings,
          file_list,
          batch_size,
          shuffle=True,
          use_multiprocess=True,
          num_workers=8,
          enable_ce=False):
    file_path = os.path.join(settings.data_dir, file_list)
    readers = []
    if 'coco' in settings.dataset:
        # cocoapi
        from pycocotools.coco import COCO
        coco_api = COCO(file_path)
        image_ids = coco_api.getImgIds()
        images = coco_api.loadImgs(image_ids)
        np.random.shuffle(images)
        if '2014' in file_list:
            sub_dir = "train2014"
        elif '2017' in file_list:
            sub_dir = "train2017"
        data_dir = os.path.join(settings.data_dir, sub_dir)
        print("data_dir:{}".format(data_dir))
        n = int(math.ceil(len(images) // num_workers)) if use_multiprocess             else len(images)
        image_lists = [images[i:i + n] for i in range(0, len(images), n)]
        for l in image_lists:
            readers.append(
                coco(settings, coco_api, l, 'train', batch_size, shuffle,
                     data_dir))
    else:
        images = [line.strip() for line in open(file_path)]
        np.random.shuffle(images)
        n = int(math.ceil(len(images) // num_workers)) if use_multiprocess             else len(images)
        image_lists = [images[i:i + n] for i in range(0, len(images), n)]
        for l in image_lists:
            readers.append(pascalvoc(settings, l, 'train', batch_size, shuffle))
    print("use_multiprocess ", use_multiprocess)
    if use_multiprocess:
        return paddle.reader.multiprocess_reader(readers, False)
    else:
        return readers[0]
    #     n = int(math.ceil(len(images) // num_workers))
    #     image_lists = [images[i:i + n] for i in range(0, len(images), n)]

    #     if '2014' in file_list:
    #         sub_dir = "train2014"
    #     elif '2017' in file_list:
    #         sub_dir = "train2017"
    #     data_dir = os.path.join(settings.data_dir, sub_dir)
    #     for l in image_lists:
    #         readers.append(
    #             coco(settings, coco_api, l, 'train', batch_size, shuffle,
    #                  data_dir))
    # else:
    #     images = [line.strip() for line in open(file_path)]
    #     n = int(math.ceil(len(images) // num_workers))
    #     image_lists = [images[i:i + n] for i in range(0, len(images), n)]
    #     for l in image_lists:
    #         readers.append(pascalvoc(settings, l, 'train', batch_size, shuffle))

    # return paddle.reader.multiprocess_reader(readers, False)


def test_data_reader(settings, file_list, batch_size):
    file_list = os.path.join(settings.data_dir, file_list)
    if 'coco' in settings.dataset:
        from pycocotools.coco import COCO
        coco_api = COCO(file_list)
        image_ids = coco_api.getImgIds()
        images = coco_api.loadImgs(image_ids)
        if '2014' in file_list:
            sub_dir = "val2014"
        elif '2017' in file_list:
            sub_dir = "val2017"
        data_dir = os.path.join(settings.data_dir, sub_dir)
        return coco(settings, coco_api, images, 'test', batch_size, False,
                    data_dir)
    else:
        image_list = [line.strip() for line in open(file_list)]
        return pascalvoc(settings, image_list, 'test', batch_size, False)


def infer(settings, image_path):
    def reader():
        if not os.path.exists(image_path):
            raise ValueError("%s is not exist, you should specify "
                             "data path correctly." % image_path)
        img = Image.open(image_path)
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


# #### 2.1.4 定义网络结构`mobilenet v1 ssd`

# In[34]:


import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr


def conv_bn(input,
            filter_size,
            num_filters,
            stride,
            padding,
            channels=None,
            num_groups=1,
            act='relu',
            use_cudnn=True,
            name=None):
    parameter_attr = ParamAttr(learning_rate=0.1, initializer=MSRA())
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        groups=num_groups,
        act=None,
        use_cudnn=use_cudnn,
        param_attr=parameter_attr,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv, act=act)


def depthwise_separable(input, num_filters1, num_filters2, num_groups, stride,
                        scale):
    depthwise_conv = conv_bn(
        input=input,
        filter_size=3,
        num_filters=int(num_filters1 * scale),
        stride=stride,
        padding=1,
        num_groups=int(num_groups * scale),
        use_cudnn=False)

    pointwise_conv = conv_bn(
        input=depthwise_conv,
        filter_size=1,
        num_filters=int(num_filters2 * scale),
        stride=1,
        padding=0)
    return pointwise_conv


def extra_block(input, num_filters1, num_filters2, num_groups, stride, scale):
    # 1x1 conv
    pointwise_conv = conv_bn(
        input=input,
        filter_size=1,
        num_filters=int(num_filters1 * scale),
        stride=1,
        num_groups=int(num_groups * scale),
        padding=0)

    # 3x3 conv
    normal_conv = conv_bn(
        input=pointwise_conv,
        filter_size=3,
        num_filters=int(num_filters2 * scale),
        stride=2,
        num_groups=int(num_groups * scale),
        padding=1)
    return normal_conv


def mobile_net(num_classes, img, img_shape, scale=1.0):
    # 300x300
    tmp = conv_bn(img, 3, int(32 * scale), 2, 1, 3)
    # 150x150
    tmp = depthwise_separable(tmp, 32, 64, 32, 1, scale)
    tmp = depthwise_separable(tmp, 64, 128, 64, 2, scale)
    # 75x75
    tmp = depthwise_separable(tmp, 128, 128, 128, 1, scale)
    tmp = depthwise_separable(tmp, 128, 256, 128, 2, scale)
    # 38x38
    tmp = depthwise_separable(tmp, 256, 256, 256, 1, scale)
    tmp = depthwise_separable(tmp, 256, 512, 256, 2, scale)

    # 19x19
    for i in range(5):
        tmp = depthwise_separable(tmp, 512, 512, 512, 1, scale)
    module11 = tmp
    tmp = depthwise_separable(tmp, 512, 1024, 512, 2, scale)

    # 10x10
    module13 = depthwise_separable(tmp, 1024, 1024, 1024, 1, scale)
    module14 = extra_block(module13, 256, 512, 1, 2, scale)
    # 5x5
    module15 = extra_block(module14, 128, 256, 1, 2, scale)
    # 3x3
    module16 = extra_block(module15, 128, 256, 1, 2, scale)
    # 2x2
    module17 = extra_block(module16, 64, 128, 1, 2, scale)

    mbox_locs, mbox_confs, box, box_var = fluid.layers.multi_box_head(
        inputs=[module11, module13, module14, module15, module16, module17],
        image=img,
        num_classes=num_classes,
        min_ratio=20,
        max_ratio=90,
        min_sizes=[60.0, 105.0, 150.0, 195.0, 240.0, 285.0],
        max_sizes=[[], 150.0, 195.0, 240.0, 285.0, 300.0],
        aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2., 3.], [2., 3.]],
        base_size=img_shape[2],
        offset=0.5,
        flip=True)

    return mbox_locs, mbox_confs, box, box_var


# #### 2.1.5 自定义网络结构`mobilenet v2 ssd`，（训练loss降不下去，效果不好）

# In[35]:


# from paddle.fluid.initializer import MSRA
# from paddle.fluid.param_attr import ParamAttr

# class MobileNetV2SSD():
#     def __init__(self, img, num_classes, img_shape, change_depth=False):
#         self.img = img
#         self.num_classes = num_classes
#         self.img_shape = img_shape
#         self.change_depth = change_depth

#     def net(self, scale=1.0):
#         change_depth = self.change_depth
#         # if change_depth is True, the new depth is 1.4 times as deep as before.
#         bottleneck_params_list = [
#             (1, 16, 1, 1),
#             (6, 24, 2, 2),
#             (6, 32, 3, 2),
#             (6, 64, 4, 2),
#             (6, 96, 3, 1),
#             (6, 160, 3, 2),
#             (6, 320, 1, 1),
#         ] if change_depth == False else [
#             (1, 16, 1, 1),
#             (6, 24, 2, 2),
#             (6, 32, 5, 2),
#             (6, 64, 7, 2),
#             (6, 96, 5, 1),
#             (6, 160, 3, 2),
#             (6, 320, 1, 1),
#         ]

#         # conv1
#         input = self.conv_bn_layer(
#             self.img,
#             num_filters=int(32 * scale),
#             filter_size=3,
#             stride=2,
#             padding=1,
#             if_act=True,
#             name='conv1_1')

#         # bottleneck sequences
#         i = 1
#         in_c = int(32 * scale)
#         module11 = None
#         for layer_setting in bottleneck_params_list:
#             t, c, n, s = layer_setting
#             i += 1
#             input = self.invresi_blocks(
#                 input=input,
#                 in_c=in_c,
#                 t=t,
#                 c=int(c * scale),
#                 n=n,
#                 s=s,
#                 name='conv' + str(i))
#             if i==6:
#                 # 19x19
#                 module11 = self.conv_bn_layer(input=input,num_filters=512,filter_size=1,stride=1,padding=0,if_act=True,name='ssd1')
#             in_c = int(c * scale)
#         # last_conv
#         tmp = self.conv_bn_layer(
#             input=input,
#             num_filters=int(1280 * scale) if scale > 1.0 else 1280,
#             filter_size=1,
#             stride=1,
#             padding=0,
#             if_act=True,
#             name='conv9')
#         module13 = self.conv_bn_layer(input=tmp,num_filters=1024,filter_size=1,stride=1,padding=0,if_act=True,name='ssd2')
#         # 10x10
#         module14 = self.extra_block(module13, 256, 512, 1, 2, scale)
#         # 5x5
#         module15 = self.extra_block(module14, 128, 256, 1, 2, scale)
#         # 3x3
#         module16 = self.extra_block(module15, 128, 256, 1, 2, scale)
#         # 2x2
#         module17 = self.extra_block(module16, 64, 128, 1, 2, scale)

#         # mbox_locs：预测的输入框的位置
#         # mbox_confs：预测框对输入的置信度
#         # box：PriorBox输出的先验框
#         # box_var：PriorBox的扩展方差
#         mbox_locs, mbox_confs, box, box_var = fluid.layers.multi_box_head(
#             inputs=[
#                 module11, module13, module14, module15, module16, module17
#             ],
#             image=self.img,
#             num_classes=self.num_classes,
#             min_ratio=20,
#             max_ratio=90,
#             min_sizes=[60.0, 105.0, 150.0, 195.0, 240.0, 285.0],
#             max_sizes=[[], 150.0, 195.0, 240.0, 285.0, 300.0],
#             aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2., 3.],
#                           [2., 3.]],
#             base_size=self.img_shape[2],
#             offset=0.5,
#             flip=True)

#         return mbox_locs, mbox_confs, box, box_var

#     def conv_bn_layer(self,
#                       input,
#                       filter_size,
#                       num_filters,
#                       stride,
#                       padding,
#                       channels=None,
#                       num_groups=1,
#                       if_act=True,
#                       name=None,
#                       use_cudnn=True):
#         conv = fluid.layers.conv2d(
#             input=input,
#             num_filters=num_filters,
#             filter_size=filter_size,
#             stride=stride,
#             padding=padding,
#             groups=num_groups,
#             act=None,
#             use_cudnn=use_cudnn,
#             param_attr=ParamAttr(name=name + '_weights'),
#             bias_attr=False)
#         bn_name = name + '_bn'
#         bn = fluid.layers.batch_norm(
#             input=conv,
#             param_attr=ParamAttr(name=bn_name + "_scale"),
#             bias_attr=ParamAttr(name=bn_name + "_offset"),
#             moving_mean_name=bn_name + '_mean',
#             moving_variance_name=bn_name + '_variance')
#         if if_act:
#             return fluid.layers.relu6(bn)
#         else:
#             return bn

#     def shortcut(self, input, data_residual):
#         return fluid.layers.elementwise_add(input, data_residual)

#     def inverted_residual_unit(self,
#                               input,
#                               num_in_filter,
#                               num_filters,
#                               ifshortcut,
#                               stride,
#                               filter_size,
#                               padding,
#                               expansion_factor,
#                               name=None):
#         num_expfilter = int(round(num_in_filter * expansion_factor))

#         channel_expand = self.conv_bn_layer(
#             input=input,
#             num_filters=num_expfilter,
#             filter_size=1,
#             stride=1,
#             padding=0,
#             num_groups=1,
#             if_act=True,
#             name=name + '_expand')

#         bottleneck_conv = self.conv_bn_layer(
#             input=channel_expand,
#             num_filters=num_expfilter,
#             filter_size=filter_size,
#             stride=stride,
#             padding=padding,
#             num_groups=num_expfilter,
#             if_act=True,
#             name=name + '_dwise',
#             use_cudnn=False)

#         linear_out = self.conv_bn_layer(
#             input=bottleneck_conv,
#             num_filters=num_filters,
#             filter_size=1,
#             stride=1,
#             padding=0,
#             num_groups=1,
#             if_act=False,
#             name=name + '_linear')
#         if ifshortcut:
#             out = self.shortcut(input=input, data_residual=linear_out)
#             return out
#         else:
#             return linear_out

#     def invresi_blocks(self, input, in_c, t, c, n, s, name=None):
#         first_block = self.inverted_residual_unit(
#             input=input,
#             num_in_filter=in_c,
#             num_filters=c,
#             ifshortcut=False,
#             stride=s,
#             filter_size=3,
#             padding=1,
#             expansion_factor=t,
#             name=name + '_1')

#         last_residual_block = first_block
#         last_c = c

#         for i in range(1, n):
#             last_residual_block = self.inverted_residual_unit(
#                 input=last_residual_block,
#                 num_in_filter=last_c,
#                 num_filters=c,
#                 ifshortcut=True,
#                 stride=1,
#                 filter_size=3,
#                 padding=1,
#                 expansion_factor=t,
#                 name=name + '_' + str(i + 1))
#         return last_residual_block

#     def extra_block(self, input, num_filters1, num_filters2, num_groups, stride,
#                     scale):
#         # 1x1 conv
#         pointwise_conv = self.conv_bn(
#             input=input,
#             filter_size=1,
#             num_filters=int(num_filters1 * scale),
#             stride=1,
#             num_groups=int(num_groups * scale),
#             padding=0)

#         # 3x3 conv
#         normal_conv = self.conv_bn(
#             input=pointwise_conv,
#             filter_size=3,
#             num_filters=int(num_filters2 * scale),
#             stride=2,
#             num_groups=int(num_groups * scale),
#             padding=1)
#         return normal_conv

#     def conv_bn(self,
#                 input,
#                 filter_size,
#                 num_filters,
#                 stride,
#                 padding,
#                 channels=None,
#                 num_groups=1,
#                 act='relu',
#                 use_cudnn=True):
#         parameter_attr = ParamAttr(learning_rate=0.1, initializer=MSRA())
#         conv = fluid.layers.conv2d(
#             input=input,
#             num_filters=num_filters,
#             filter_size=filter_size,
#             stride=stride,
#             padding=padding,
#             groups=num_groups,
#             act=None,
#             use_cudnn=use_cudnn,
#             param_attr=parameter_attr,
#             bias_attr=False)
#         return fluid.layers.batch_norm(input=conv, act=act)

# def MobileNetV2_x0_25(img, num_classes, img_shape):
#     model = MobileNetV2SSD(img, num_classes, img_shape)
#     return model.net(scale=0.25)


# def MobileNetV2_x0_5(img, num_classes, img_shape):
#     model = MobileNetV2SSD(img, num_classes, img_shape)
#     return model.net(scale=0.5)


# def MobileNetV2_x1_0(img, num_classes, img_shape):
#     model = MobileNetV2SSD(img, num_classes, img_shape)
#     return model.net(scale=1.0)


# def MobileNetV2_x1_5(img, num_classes, img_shape):
#     model = MobileNetV2SSD(img, num_classes, img_shape)
#     return model.net(scale=1.5)


# def MobileNetV2_x2_0(img, num_classes, img_shape):
#     model = MobileNetV2SSD(img, num_classes, img_shape)
#     return model.net(scale=2.0)


# def MobileNetV2_scale(img, num_classes, img_shape):
#     model = MobileNetV2SSD(img, num_classes, img_shape, change_depth=True)
#     return model.net(scale=1.2)


# #### 2.1.6 定义部分训练用到的参数

# In[36]:


import os
import time
import numpy as np
import argparse
import functools
import shutil
import math
import multiprocessing

import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib.slim import Compressor

args_learning_rate= 0.00000001
# args_learning_rate= 0.000005
# args_learning_rate = 0.001
args_batch_size = 32
args_epoc_num = 120
args_use_gpu = True
args_parallel = True
args_use_multiprocess = True

#保存save_persistables模型的路径
args_model_save_dir = '/home/aistudio/work/models/quant_models'


train_parameters = {
    "pascalvoc": {
        "train_images": 16551,
        "image_shape": [3, 300, 300],
        "class_num": 21,
        "batch_size": 64,
        "lr": 0.001,
        "lr_epochs": [40, 60, 80, 100],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01],
        "ap_version": '11point',
    },
    "coco2014": {
        "train_images": 82783,
        "image_shape": [3, 300, 300],
        "class_num": 81,
        "batch_size": 64,
        "lr": 0.001,
        "lr_epochs": [12, 19],
        "lr_decay": [1, 0.5, 0.25],
        "ap_version": 'integral', # should use eval_coco_map.py to test model
    },
    "coco2017": {
        "train_images": 118287,
        "image_shape": [3, 300, 300],
        "class_num": 81,
        "batch_size": 64,
        "lr": 0.0000001,
        "lr_epochs": [12, 19],
        "lr_decay": [1, 0.5, 0.25],
        "ap_version": 'integral', # should use eval_coco_map.py to test model
    }
}

train_parameters[dataset]['image_shape'] = image_shape
train_parameters[dataset]['batch_size'] = args_batch_size
train_parameters[dataset]['lr'] = args_learning_rate
train_parameters[dataset]['epoc_num'] = args_epoc_num
train_parameters[dataset]['ap_version'] = args_ap_version


# #### 2.1.7 设置优化器策略

# In[37]:


# def optimizer_setting(train_params):
#     batch_size = train_params["batch_size"]
#     iters = train_params["train_images"] // batch_size
#     lr = train_params["lr"]
#     boundaries = [i * iters  for i in train_params["lr_epochs"]]
#     values = [ i * lr for i in train_params["lr_decay"]]

#     optimizer = fluid.optimizer.RMSProp(
#         learning_rate=fluid.layers.piecewise_decay(boundaries, values),
#         # learning_rate=args_learning_rate,
#         # learning_rate=0.1,
#         regularization=fluid.regularizer.L2Decay(0.00005), )

#     return optimizer

def optimizer_setting(train_params):
    batch_size = train_params["batch_size"]
    boundaries=[train_params["train_images"] / batch_size * 10,
                    train_params["train_images"] / batch_size * 16]
    values=[1e-4, 1e-5, 1e-6]
    opt = fluid.optimizer.Momentum(
        momentum=0.9,
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=boundaries,
            values=values),
        regularization=fluid.regularizer.L2Decay(4e-5))
    return opt


# #### 2.1.8 修改ssd_loss函数，替换原有框架中ssd_loss函数
# 主要拆分其中nn.softmax_with_cross_entropy函数为nn.softmax和nn.cross_entropy两个函数，避免后续量化、剪枝操作后的训练中，loss为NAN的错误。

# In[38]:


from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers import nn, iou_similarity, bipartite_match, target_assign, tensor, box_coder


def ssd_loss(location,
             confidence,
             gt_box,
             gt_label,
             prior_box,
             prior_box_var=None,
             background_label=0,
             overlap_threshold=0.5,
             neg_pos_ratio=3.0,
             neg_overlap=0.5,
             loc_loss_weight=1.0,
             conf_loss_weight=1.0,
             match_type='per_prediction',
             mining_type='max_negative',
             normalize=True,
             sample_size=None):

    helper = LayerHelper('ssd_loss', **locals())
    if mining_type != 'max_negative':
        raise ValueError("Only support mining_type == max_negative now.")

    num, num_prior, num_class = confidence.shape
    conf_shape = nn.shape(confidence)

    def __reshape_to_2d(var):
        return nn.flatten(x=var, axis=2)

    # 1. Find matched boundding box by prior box.
    #   1.1 Compute IOU similarity between ground-truth boxes and prior boxes.
    iou = iou_similarity(x=gt_box, y=prior_box)
    #   1.2 Compute matched boundding box by bipartite matching algorithm.
    matched_indices, matched_dist = bipartite_match(iou, match_type,
                                                    overlap_threshold)

    # 2. Compute confidence for mining hard examples
    # 2.1. Get the target label based on matched indices
    gt_label = nn.reshape(
        x=gt_label, shape=(len(gt_label.shape) - 1) * (0, ) + (-1, 1))
    gt_label.stop_gradient = True
    target_label, _ = target_assign(
        gt_label, matched_indices, mismatch_value=background_label)
    # 2.2. Compute confidence loss.
    # Reshape confidence to 2D tensor.
    confidence = __reshape_to_2d(confidence)
    target_label = tensor.cast(x=target_label, dtype='int64')
    target_label = __reshape_to_2d(target_label)
    target_label.stop_gradient = True
    #conf_loss = nn.softmax_with_cross_entropy(confidence, target_label)
    conf_softmax = nn.softmax(confidence,use_cudnn=False)
    conf_loss = nn.cross_entropy(conf_softmax, target_label)
    # 3. Mining hard examples
    actual_shape = nn.slice(conf_shape, axes=[0], starts=[0], ends=[2])
    actual_shape.stop_gradient = True
    conf_loss = nn.reshape(
        x=conf_loss, shape=(num, num_prior), actual_shape=actual_shape)
    conf_loss.stop_gradient = True
    neg_indices = helper.create_variable_for_type_inference(dtype='int32')
    dtype = matched_indices.dtype
    updated_matched_indices = helper.create_variable_for_type_inference(
        dtype=dtype)
    helper.append_op(
        type='mine_hard_examples',
        inputs={
            'ClsLoss': conf_loss,
            'LocLoss': None,
            'MatchIndices': matched_indices,
            'MatchDist': matched_dist,
        },
        outputs={
            'NegIndices': neg_indices,
            'UpdatedMatchIndices': updated_matched_indices
        },
        attrs={
            'neg_pos_ratio': neg_pos_ratio,
            'neg_dist_threshold': neg_overlap,
            'mining_type': mining_type,
            'sample_size': sample_size,
        })

    # 4. Assign classification and regression targets
    # 4.1. Encoded bbox according to the prior boxes.
    encoded_bbox = box_coder(
        prior_box=prior_box,
        prior_box_var=prior_box_var,
        target_box=gt_box,
        code_type='encode_center_size')
    # 4.2. Assign regression targets
    target_bbox, target_loc_weight = target_assign(
        encoded_bbox, updated_matched_indices, mismatch_value=background_label)
    # 4.3. Assign classification targets
    target_label, target_conf_weight = target_assign(
        gt_label,
        updated_matched_indices,
        negative_indices=neg_indices,
        mismatch_value=background_label)

    # 5. Compute loss.
    # 5.1 Compute confidence loss.
    target_label = __reshape_to_2d(target_label)
    target_label = tensor.cast(x=target_label, dtype='int64')

    # conf_loss = nn.softmax_with_cross_entropy(confidence, target_label)
    conf_softmax = nn.softmax(confidence, use_cudnn=False)
    conf_loss = nn.cross_entropy(conf_softmax, target_label)
    target_conf_weight = __reshape_to_2d(target_conf_weight)
    conf_loss = conf_loss * target_conf_weight

    # the target_label and target_conf_weight do not have gradient.
    target_label.stop_gradient = True
    target_conf_weight.stop_gradient = True

    # 5.2 Compute regression loss.
    location = __reshape_to_2d(location)
    target_bbox = __reshape_to_2d(target_bbox)

    loc_loss = nn.smooth_l1(location, target_bbox)
    target_loc_weight = __reshape_to_2d(target_loc_weight)
    loc_loss = loc_loss * target_loc_weight

    # the target_bbox and target_loc_weight do not have gradient.
    target_bbox.stop_gradient = True
    target_loc_weight.stop_gradient = True

    # 5.3 Compute overall weighted loss.
    loss = conf_loss_weight * conf_loss + loc_loss_weight * loc_loss
    # reshape to [N, Np], N is the batch size and Np is the prior box number.
    loss = nn.reshape(x=loss, shape=(num, num_prior), actual_shape=actual_shape)
    loss = nn.reduce_sum(loss, dim=1, keep_dim=True)
    if normalize:
        normalizer = nn.reduce_sum(target_loc_weight)
        loss = loss / normalizer

    return loss


# #### 2.1.9 定义执行程序用到的`program`

# In[40]:


def build_program(main_prog, startup_prog, train_params, is_train):
    image_shape = train_params['image_shape']
    class_num = train_params['class_num']
    ap_version = train_params['ap_version']
    outs = []
    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=64,
            shapes=[[-1] + image_shape, [-1, 4], [-1, 1], [-1, 1]],
            lod_levels=[0, 1, 1, 1],
            dtypes=["float32", "float32", "int32", "int32"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, gt_box, gt_label, difficult = fluid.layers.read_file(py_reader)
            locs, confs, box, box_var = mobile_net(class_num, image, image_shape)
            # locs, confs, box, box_var = MobileNetV2_x1_0(image, class_num, image_shape)
            boxes = fluid.layers.box_coder(
                prior_box=box,
                prior_box_var=box_var,
                target_box=locs,
                code_type='decode_center_size')
            scores = fluid.layers.nn.softmax(input=confs)
            scores = fluid.layers.nn.transpose(scores, perm=[0, 2, 1])
            scores.stop_gradient = True
            
            inference_fetch_list = [boxes] + [scores]
            # print("inference_fetch_list: {}".format(inference_fetch_list))
            gt_label.stop_gradient=True
            difficult.stop_gradient=True
            gt_box.stop_gradient=True
            if is_train:
                with fluid.unique_name.guard("train"):
                    loss = ssd_loss(locs, confs, gt_box, gt_label, box,
                        box_var)
                    loss = fluid.layers.reduce_sum(loss)
                    optimizer = optimizer_setting(train_params)
                    optimizer.minimize(loss)
                    # print("loss: {}".format(loss))
                outs = [py_reader,(image, gt_box, gt_label, difficult), loss, optimizer, inference_fetch_list]
            else:
                with fluid.unique_name.guard("inference"):
                    # nmsed_out = fluid.layers.detection_output(
                    #     locs, confs, box, box_var, nms_top_k=-1, nms_threshold=0.45, keep_top_k=-1)#输出一个LoDTensor，形为[No,6]。每行有6个值：[label,confidence,xmin,ymin,xmax,ymax]
                    # confs = fluid.layers.transpose(confs, perm=[0, 2, 1])
                    decoded_box = fluid.layers.box_coder(
                        prior_box=box,
                        prior_box_var=box_var,
                        target_box=locs,
                        code_type='decode_center_size')
                    confs = fluid.layers.softmax(input=confs)
                    confs = fluid.layers.transpose(confs, perm=[0, 2, 1])
                    confs.stop_gradient = True
                    nmsed_out = fluid.layers.multiclass_nms(
                        bboxes=decoded_box,
                        scores=confs,
                        score_threshold=0.01,
                        nms_top_k=-1,
                        nms_threshold=0.45,
                        keep_top_k=-1,
                        normalized=False)
                    # print("nmsed_out: {}".format(nmsed_out))
                    gt_label = fluid.layers.cast(x=gt_label, dtype=gt_box.dtype)
                    if difficult:
                        difficult = fluid.layers.cast(x=difficult, dtype=gt_box.dtype)
                        gt_label = fluid.layers.reshape(gt_label, [-1, 1])
                        difficult = fluid.layers.reshape(difficult, [-1, 1])
                        label = fluid.layers.concat([gt_label, difficult, gt_box], axis=1)
                    else:
                        label = fluid.layers.concat([gt_label, gt_box], axis=1)
                    map_eval = fluid.layers.detection.detection_map(
                        nmsed_out,
                        label,
                        class_num,
                        background_label=0,
                        overlap_threshold=0.5,
                        evaluate_difficult=False,
                        ap_version=ap_version)
                    # print("map_eval: {}".format(map_eval))
                # nmsed_out and image is used to save mode for inference
                outs = [py_reader, (image, gt_box, gt_label, difficult), map_eval, nmsed_out, image]
    return outs


# #### 2.1.10 定义部分变量参数

# In[42]:


model_save_dir = args_model_save_dir
pretrained_model = args_pretrained_model
use_gpu = args_use_gpu
parallel = args_parallel
is_shuffle = True
use_multiprocess = args_use_multiprocess


# #### 2.1.11 定义训练、测试的`program`、`reader`等

# In[43]:


if not use_gpu:
    devices_num = int(os.environ.get('CPU_NUM',
                          multiprocessing.cpu_count()))
    # devices_num = 1
else:
    devices_num = fluid.core.get_cuda_device_count()

batch_size = train_parameters[dataset]['batch_size']
epoc_num = train_parameters[dataset]['epoc_num']

batch_size_per_device = batch_size // devices_num 

startup_prog = fluid.Program()
train_prog = fluid.Program()
test_prog = fluid.Program()

train_py_reader, train_inputs, loss, optimizer, inference_fetch_list = build_program(
        main_prog=train_prog,
        startup_prog=startup_prog,
        train_params=train_parameters[dataset],
        is_train=True)
test_py_reader, test_inputs, map_var, nmsed_out, image = build_program(
        main_prog=test_prog,
        startup_prog=startup_prog,
        train_params=train_parameters[dataset],
        is_train=False)

# train_inputs, loss, optimizer = build_program(
#         main_prog=train_prog,
#         startup_prog=startup_prog,
#         train_params=train_parameters[dataset],
#         is_train=True)
    
# test_inputs, map_var, _, _ = build_program(
#         main_prog=test_prog,
#         startup_prog=startup_prog,
#         train_params=train_parameters[dataset],
#         is_train=False)
    
test_prog = test_prog.clone(for_test=True) 


# #### 2.1.12 定义训练环境相关参数

# In[44]:


place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_prog)
    
if pretrained_model:
    def if_exist(var):
        return os.path.exists(os.path.join(pretrained_model, var.name))
    fluid.io.load_vars(exe, pretrained_model, main_program=train_prog,
                           predicate=if_exist)

# for param in train_prog.global_block().all_parameters():
    # print("param name: {}".format(param.name))
    # print("name: {}; shape: {}".format(param.name, param.shape))

# if parallel:
#     loss.persistable = True
#     build_strategy = fluid.BuildStrategy()
#     build_strategy.enable_inplace = True
#     build_strategy.memory_optimize = True
#     train_exe = fluid.ParallelExecutor(main_program=train_prog,
#         use_cuda=use_gpu, loss_name=loss.name, build_strategy=build_strategy)


# #### 2.1.13 获取训练、测试数据

# In[45]:


get_ipython().system('pip install pycocotools ')
num_workers = 8
train_reader = train_data_reader(data_args,
                                train_file_list,
                                batch_size_per_device,
                                shuffle=is_shuffle,
                                use_multiprocess=use_multiprocess,
                                num_workers=num_workers)
train_py_reader.decorate_paddle_reader(train_reader)

test_reader = test_data_reader(data_args, val_file_list, batch_size)
test_py_reader.decorate_paddle_reader(test_reader)


# #### 2.1.14 定义训练、测试的输入、输出

# In[46]:


image, gt_box, gt_label, difficult = train_inputs
# train_feed_list = [("image", "image"), ("gt_box", "gt_box"), ("gt_label", "gt_label"), ("difficult", "difficult")]
train_feed_list = [("image", image.name), ("gt_box", gt_box.name), ("gt_label", gt_label.name), ("difficult", difficult.name)]
train_fetch_list=[("loss", loss.name)]

image, gt_box, gt_label, difficult = test_inputs
# val_feed_list=[("image", "image"), ("gt_box", "gt_box"), ("gt_label", "gt_label"), ("difficult", "difficult")]
val_feed_list = [("image", image.name), ("gt_box", gt_box.name), ("gt_label", gt_label.name), ("difficult", difficult.name)]
val_fetch_list=[("map",  map_var.name)]


# #### 2.1.15 定义剪枝、量化配置策略的配置文件

# In[47]:


# -*- coding: UTF-8 -*-
#先裁剪，再量化配置
config="""
version: 1.0
pruners:
    pruner_1:
        class: 'StructurePruner'
        pruning_axis:
            '*': 0
        criterions:
            '*': 'l1_norm'
strategies:
    uniform_pruning_strategy:
        class: 'UniformPruneStrategy'
        pruner: 'pruner_1'
        start_epoch: 0
        target_ratio: 0.05
        # pruned_params: 'conv2d_2.w_0|conv2d_4.w_0|conv2d_[6-9].w_0|conv2d_1[0-1].w_0'
        pruned_params: 'conv2d_[5-6].w_0'
        metric_name: 'map'
    quantization_strategy:
        class: 'QuantizationStrategy'
        # start_epoch: 121
        # end_epoch: 141
        start_epoch: 1
        end_epoch: 2
        float_model_save_path: '/home/aistudio/work/models/output/float'
        mobile_model_save_path: '/home/aistudio/work/models/output/mobile'
        int8_model_save_path: '/home/aistudio/work/models/output/int8'
        weight_bits: 8
        activation_bits: 8
        weight_quantize_type: 'abs_max'
        activation_quantize_type: 'abs_max'
        # save_in_nodes: ['image']
        # save_out_nodes: ['inferenceinferencedetection_output_0.tmp_0']
compressor:
    # epoch: 142
    epoch: 3
    # Please enable this option for loading checkpoint.
    # init_model: '/home/aistudio/work/models/checkpoints_1/7/'
    checkpoint_path: '/home/aistudio/work/models/checkpoints/'
    strategies:
        - uniform_pruning_strategy
        - quantization_strategy
"""

f = open("/home/aistudio/work/models/compress.yaml", 'w')
f.write(config)
f.close()


# In[48]:


# # -*- coding: UTF-8 -*-
# #先裁剪，再量化配置
# config="""
# version: 1.0
# pruners:
#     pruner_1:
#         class: 'StructurePruner'
#         pruning_axis:
#             '*': 0
#         criterions:
#             '*': 'l1_norm'
# strategies:
#     uniform_pruning_strategy:
#         class: 'UniformPruneStrategy'
#         pruner: 'pruner_1'
#         start_epoch: 0
#         target_ratio: 0.50
#         pruned_params: '.*_sep_weights'
#         metric_name: 'map'
#     quantization_strategy:
#         class: 'QuantizationStrategy'
#         # start_epoch: 121
#         # end_epoch: 141
#         start_epoch: 12
#         end_epoch: 13
#         float_model_save_path: '/home/aistudio/work/models/output/float'
#         mobile_model_save_path: '/home/aistudio/work/models/output/mobile'
#         int8_model_save_path: '/home/aistudio/work/models/output/int8'
#         weight_bits: 8
#         activation_bits: 8
#         weight_quantize_type: 'abs_max'
#         activation_quantize_type: 'abs_max'
#         # save_in_nodes: ['image']
#         # save_out_nodes: ['inferenceinferencedetection_output_0.tmp_0']
# compressor:
#     # epoch: 142
#     epoch: 14
#     # Please enable this option for loading checkpoint.
#     # init_model: '/home/aistudio/work/models/checkpoints/4/'
#     checkpoint_path: '/home/aistudio/work/models/checkpoints/'
#     strategies:
#         - uniform_pruning_strategy
#         - quantization_strategy
# """

# f = open("/home/aistudio/work/models/compress.yaml", 'w')
# f.write(config)
# f.close()


# In[19]:


# # -*- coding: UTF-8 -*-
# #量化配置
# config="""
# version: 1.0
# strategies:
#     quantization_strategy:
#         class: 'QuantizationStrategy'
#         start_epoch: 0
#         end_epoch: 1
#         float_model_save_path: '/home/aistudio/work/models/output/float'
#         mobile_model_save_path: '/home/aistudio/work/models/output/mobile'
#         int8_model_save_path: '/home/aistudio/work/models/output/int8'
#         weight_bits: 8
#         activation_bits: 8
#         weight_quantize_type: 'abs_max'
#         activation_quantize_type: 'abs_max'
#         # save_in_nodes: ['image']
#         # save_out_nodes: ['inferenceinferencedetection_output_0.tmp_0']
# compressor:
#     epoch: 2
#     checkpoint_path: '/home/aistudio/work/models/checkpoints/'
#     strategies:
#         - quantization_strategy
# """

# f = open("/home/aistudio/work/models/compress.yaml", 'w')
# f.write(config)
# f.close()


# In[20]:


# # -*- coding: UTF-8 -*-
# #剪裁配置
# config="""
# version: 1.0
# pruners:
#     pruner_1:
#         class: 'StructurePruner'
#         pruning_axis:
#             '*': 0
#         criterions:
#             '*': 'l1_norm'
# strategies:
#     uniform_pruning_strategy:
#         class: 'UniformPruneStrategy'
#         pruner: 'pruner_1'
#         start_epoch: 0
#         target_ratio: 0.5
#         pruned_params: '.*_sep_weights'
#         metric_name: 'map'
# compressor:
#     epoch: 2
#     #init_model: './checkpoints/0' # Please enable this option for loading checkpoint.
#     checkpoint_path: '/home/aistudio/work/models/checkpoints/'
#     strategies:
#         - uniform_pruning_strategy
# """

# f = open("/home/aistudio/work/models/compress.yaml", 'w')
# f.write(config)
# f.close()


# #### 2.1.16 执行剪枝、量化训练

# In[24]:


# -*- coding: UTF-8 -*-
com_pass = Compressor(
        place,
        fluid.global_scope(),
        train_prog,
        train_reader=train_py_reader,
        # train_feed_list=train_feed_list,
        train_feed_list=None,
        train_fetch_list=train_fetch_list,
        eval_program=test_prog,
        eval_reader=test_py_reader,
        # eval_feed_list=val_feed_list,
        eval_feed_list=None,
        eval_fetch_list=val_fetch_list,
        train_optimizer=None)
com_pass.config('/home/aistudio/work/models/compress.yaml')
eval_graph = com_pass.run()


# #### 2.1.17 保存剪枝、量化后的模型

# In[25]:


peuned_program = com_pass.eval_graph.program
# for param in peuned_program.global_block().all_parameters():
#     print("param name: {}".format(param.name))
fluid.io.save_inference_model("/home/aistudio/work/models/pruned_models/20190915_best_1", feeded_var_names=[image.name], target_vars=inference_fetch_list, executor=exe, main_program=peuned_program)
fluid.io.save_persistables(exe, model_save_dir + "/20190915_best_1", main_program=peuned_program)


# #### 2.1.18 计算分数

# In[26]:


#计算分数
get_ipython().system('/bin/bash /home/aistudio/work/astar2019/testssd.sh')


# #### 2.1.19 提交模型

# In[30]:


#提交模型
# !zip -r /home/aistudio/work/models_20190804/submit/submit_model.zip /home/aistudio/work/models/pruned_models/20190915_best
# !cd /home/aistudio/work/models_20190804/submit && rm -rf submit.sh && wget -O submit.sh http://ai-studio-static.bj.bcebos.com/script/submit.sh && sh submit.sh /home/aistudio/work/models_20190804/submit/submit_model.zip bb76abd7312b4bf28aa5770c0a57053f

