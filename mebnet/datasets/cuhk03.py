from __future__ import print_function, absolute_import
import numpy as np
import pdb
from glob import glob
import re
import os


class CUHK03(object):

    def __init__(self, root):

        self.images_dir = os.path.join(root, 'cuhk03-np/detected')
        self.train_path = 'bounding_box_train'
        self.gallery_path = 'bounding_box_test'
        self.query_path = 'query'
#        self.camstyle_path = 'bounding_box_train_camstyle'
        self.train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0
        self.load()

    def preprocess(self, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(os.path.join(self.images_dir, path, '*.png')))
        for fpath in fpaths:
            fname = os.path.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            ret.append((os.path.join(path, fname), pid, cam))
        return ret, int(len(all_pids))

    def load(self):
        self.train, self.num_train_pids = self.preprocess(self.train_path)
        self.gallery, self.num_gallery_pids = self.preprocess(self.gallery_path, False)
        self.query, self.num_query_pids = self.preprocess(self.query_path, False)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_pids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_pids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_pids, len(self.gallery)))
