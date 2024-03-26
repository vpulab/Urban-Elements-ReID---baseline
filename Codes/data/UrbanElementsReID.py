# encoding: utf-8
"""
@author:  weijian
@contact: dengwj16@gmail.com
"""

import glob
import csv
import re
import torch
import xml.dom.minidom as XD
import os.path as osp
import xml.etree.ElementTree as ET

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
@DATASET_REGISTRY.register()


class UrbanElementsReID(ImageDataset):

    def __init__(self, root='/home/jgf/Desktop/rhome/jgf/baselineChallenge/UrbanElementsReID',
                 verbose=True, **kwargs):
        self.dataset_dir = root

        self.train_dir = osp.join(self.dataset_dir, 'image_train/')
        self.query_dir = osp.join(self.dataset_dir, 'image_train/')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_train/')

        self._check_before_run()
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        self.train = train
        self.query = query
        self.gallery = gallery

        super(UrbanElementsReID, self).__init__(self.train , self.query, self.gallery, **kwargs)

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
        pids = []
        with open(csv_dir, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                camids.append(row[0])
                imageNames.append(str(row[1]))
                pids.append(-1)
        
        return list(zip(camids, imageNames, pids))
    
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
    
    def _process_dir_test(self, dir_path, relabel=False, query=True):
        
        dataset = []

        if query:
            xml_dir = osp.join(self.dataset_dir_test, 'query.csv')
        else:
            xml_dir = osp.join(self.dataset_dir_test, 'test.csv')

        xml_file = self._readCSV_eval_(xml_dir)
        
        for camid, imageName, pid in xml_file:
            camid = int(camid[1:])
            dataset.append((osp.join(dir_path, imageName), pid, camid))
                
        return dataset 

    def _process_track(self, path): 
        
        file = open(path)
        tracklet = dict()
        frame2trackID = dict()
        nums = []
        
        for track_id, line in enumerate(file.readlines()):
            curLine = line.strip().split(" ")
            nums.append(len(curLine))
            tracklet[track_id] =  curLine
            for frame in curLine:
                frame2trackID[frame] = track_id
                
        return tracklet, nums, frame2trackID

    def _process_dir_testVeri(self, dir_path, relabel=False):
        
        dataset = []
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d\d\d)')
        pid_container = set()
        
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
