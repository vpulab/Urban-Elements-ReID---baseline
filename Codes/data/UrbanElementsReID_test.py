# encoding: utf-8
"""
@author:  weijian
@contact: dengwj16@gmail.com
"""

import glob
import re
import csv
import xml.dom.minidom as XD
import os.path as osp
import xml.etree.ElementTree as ET

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
from ..datasets import DATASET_REGISTRY
@DATASET_REGISTRY.register()

    

class UrbanElementsReID_test(ImageDataset):

    def __init__(self, root='/home/jgf/Desktop/rhome/jgf/baselineChallenge/UrbanElementsReID',
                 verbose=True, **kwargs):
        self.dataset_dir = root
        self.dataset_dir_test = root

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train/')
        self.query_dir = osp.join(self.dataset_dir_test, 'image_query/')
        self.gallery_dir = osp.join(self.dataset_dir_test, 'image_test/')

        self._check_before_run()
        query = self._process_dir_test(self.query_dir, relabel=False,query=True)
        gallery = self._process_dir_test(self.gallery_dir, relabel=False)
        train = self._process_dir(self.train_dir, relabel=True)

        self.train = train
        self.query = query
        self.gallery = gallery

        super(UrbanElementsReID_test, self).__init__(self.train , self.query, self.gallery, **kwargs)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _readCSV_(self, csv_dir):
        
        camids = []
        imageNames = []
        pids = []
        with open(csv_dir, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                camids.append(row[0])
                imageNames.append(str(row[1]))
                pids.append(int(row[2]))
        
        return list(zip(camids, imageNames, pids))

    def _readCSV_eval_(self, csv_dir):
        camids = []
        imageNames = []
        with open(csv_dir, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                camids.append(row[0])
                imageNames.append(str(row[1]))
        
        return list(zip(camids, imageNames))
    
    def _process_dir(self, dir_path, relabel=False):
        xml_dir = osp.join(self.dataset_dir, 'train.csv')
        xml_file = self._readCSV_(xml_dir)

        pid_container = set()

        for _, _, pid in xml_file:
            if pid == -1: continue
            pid_container.add(pid)       
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        dataset = []

        for camid, imageName, pid in xml_file:
            camid = int(camid[1:])
            if pid == -1: continue
            if relabel: pid = pid2label[pid]
            dataset.append((osp.join(dir_path, imageName), pid, camid))
                
        return dataset

    def _process_dir_test(self, dir_path, relabel=False, query=False):
        
        dataset = []
        if query:
            xml_dir = osp.join(self.dataset_dir_test, 'query.csv')
        else:
            xml_dir = osp.join(self.dataset_dir_test, 'test.csv')
            
        xml_file = self._readCSV_eval_(xml_dir)
        
        for cid, imageName in xml_file:
            camid = int(cid[1:])
            dataset.append((osp.join(dir_path, imageName), -1, camid))
                
        return dataset
    
    def _process_dir_demo(self, dir_path, relabel=False):
        
        dataset = []
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.sort()
        pid_container = set()
        
        for img_path in img_paths:
            pid = 1
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        for img_path in img_paths:
            pid, camid = 1, 2
            if pid == -1: continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
