import numpy as np
from torch.utils import data as data

from basicsr.data.bsrgan_util import degradation_bsrgan
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

from .data_util import make_dataset
from .bsrgan_light_util import degradation_bsrgan_light

import cv2
import random
import csv
import pandas as pd
import os


def random_resize(img, scale_factor=1.):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)


def random_crop(img, out_size):
    h, w = img.shape[:2]
    rnd_h = random.randint(0, h - out_size)
    rnd_w = random.randint(0, w - out_size)
    return img[rnd_h: rnd_h + out_size, rnd_w: rnd_w + out_size]


@DATASET_REGISTRY.register()
class BSRGANTrainDataset(data.Dataset):
    """Synthesize LR-HR pairs online with BSRGAN for image restoration.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(BSRGANTrainDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
                
        self.gt_paths = make_dataset(self.gt_folder)

        # text_labels_list = [row for row in csv.reader(open(opt['text_label_file']))]
        
        # self.gt_paths = [x for x in self.gt_paths if 'OST_train_HR_full' in x]
        # self.gt_paths = [x for x in self.gt_paths if 'OST_train_HR_sub' not in x]

        # self.text_labels = {}
        # for it in text_labels_list:
        #     self.text_labels[it[0]] = it[1]

    def __getitem__(self, index):
        
        scale = self.opt['scale']

        gt_path = self.gt_paths[index]
        img_gt = cv2.imread(gt_path)

        while img_gt is None:
            index = random.randint(0, len(self.gt_paths) - 1)
            gt_path = self.gt_paths[index]
            img_gt = cv2.imread(gt_path)
        
        img_gt = img_gt.astype(np.float32) / 255.

        img_gt = img_gt[:, :, [2, 1, 0]] # BGR to RGB
        gt_size = self.opt['gt_size']

        # gt_basename = os.path.basename(gt_path)
        # key = gt_basename[:-9] + gt_basename[-4:]
        # if key in self.text_labels.keys(): 
        #     text = self.text_labels[key]
        # else:
        #     text = 'face' 

        if self.opt['phase'] == 'train':
            if self.opt.get('use_resize_crop', False):
                input_gt_size = img_gt.shape[0]
                input_gt_random_size = random.randint(gt_size, input_gt_size)
                resize_factor = input_gt_random_size / input_gt_size
                img_gt = random_resize(img_gt, resize_factor)
            
            if self.opt.get('use_resize', False):
                input_gt_size = min(img_gt.shape[0], img_gt.shape[1])
                resize_factor = gt_size / input_gt_size
                img_gt = random_resize(img_gt, resize_factor)
            
            img_gt = random_crop(img_gt, gt_size)

        img_lq, img_gt = degradation_bsrgan_light(img_gt, sf=scale, lq_patchsize=self.opt['gt_size'] // scale, use_crop=False)
        img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path,
            # 'text': text,
        }

    def __len__(self):
        return len(self.gt_paths)
