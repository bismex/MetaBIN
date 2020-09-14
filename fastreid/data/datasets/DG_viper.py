# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from glob import glob

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['DG_viper', ]


@DATASET_REGISTRY.register()
class DG_VIPeR(ImageDataset):
    dataset_dir = "viper"
    dataset_name = "viper"

    def __init__(self, root='datasets', **kwargs):
        if isinstance(root, list):
            type = root[1]
            self.root = root[0]
        else:
            self.root = root
            type = 'split_1a'
        self.train_dir = os.path.join(self.root, self.dataset_dir, type, 'train')
        self.query_dir = os.path.join(self.root, self.dataset_dir, type, 'query')
        self.gallery_dir = os.path.join(self.root, self.dataset_dir, type, 'gallery')

        required_files = [
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_train(self.train_dir, is_train = True)
        query = self.process_train(self.query_dir, is_train = False)
        gallery = self.process_train(self.gallery_dir, is_train = False)

        super().__init__(train, query, gallery, **kwargs)

    def process_train(self, path, is_train = True):
        data = []
        img_list = glob(os.path.join(path, '*.png'))
        for img_path in img_list:
            img_name = img_path.split('/')[-1] # p000_c1_d045.png
            split_name = img_name.split('_')
            pid = int(split_name[0][1:])
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
            camid = int(split_name[1][1:])
            # dirid = int(split_name[2][1:-4])
            data.append([img_path, pid, camid])

        return data