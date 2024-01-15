#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File : myData.py
# @Author : DingjieFu
# @Time : 2024/01/09 15:38:51
"""
    Prepare&Process Dataset
"""

import torch
from torch.utils.data.dataset import Dataset

import os
import random
import scipy.io
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ZSLDataset(Dataset):
    def __init__(self, dset, n_train, n_test, gzsl=False, is_train=True, synthetic=False, syn_dataset=None):
        '''
            - dset: Name of dataset - among [sun, cub, awa1, awa2]
            - n_train: Number of train classes
            - n_test: Number of test classes
            - gzsl: Boolean for Generalized ZSL
            - is_train: Boolean indicating whether train/test
            - synthetic: Boolean indicating whether dataset is for synthetic examples
            - syn_dataset: A list consisting of 3-tuple (z, _, y) used for sampling only when synthetic flag is True
        '''
        super(ZSLDataset, self).__init__()
        self.dset = dset
        self.n_train = n_train
        self.n_test = n_test
        self.train = is_train
        self.gzsl = gzsl
        self.synthetic = synthetic

        # feature
        res101 = scipy.io.loadmat('%s/res101.mat' % dset)
        # (sample_nums x feature_dims)
        self.features = self.normalize(res101['features'].T)
        self.labels = res101['labels'].reshape(-1)

        # attribute
        self.att_splits = scipy.io.loadmat('%s/att_splits.mat' % dset)
        # (class_nums x attributes_dims)
        self.attributes = self.att_splits['att'].T

        self.get_classData()

        # whether dataset is for synthetic examples
        if self.synthetic:
            assert syn_dataset is not None
            self.syn_dataset = syn_dataset
        else:
            self.dataset = self.create_orig_dataset()
            if self.train:
                self.gzsl_dataset = self.create_gzsl_dataset()

    def normalize(self, matrix):
        """
            - Normalize data
        """
        scaler = MinMaxScaler()
        return scaler.fit_transform(matrix)

    def get_classData(self):
        '''
            - get -> sample index | class label | class feature
        '''
        # sample index
        self.trainval_index = self.att_splits['trainval_loc'].reshape(-1)
        self.test_index = self.att_splits['test_unseen_loc'].reshape(-1)
        # sample label(-1 -> self.labels is a array, to get its idx)
        # (train_samples x 1)
        self.labels_trainval = self.labels[self.trainval_index-1]
        # (test_samples x 1)
        self.labels_test = self.labels[self.test_index-1]
        # class label(unique)
        self.trainval_labels = np.unique(self.labels_trainval)
        self.test_labels = np.unique(self.labels_test)
        # class feature
        # (train_samples x feature_dims)
        self.trainval_feats = self.features[self.trainval_index-1]
        # (test_samples x feature_dims)
        self.test_feats = self.features[self.test_index-1]

    def create_gzsl_dataset(self, n_samples=200):
        '''
        Create an auxillary dataset to be used during training final
        classifier on seen classes
        '''
        dataset = []
        for key in self.gzsl_map.keys():
            features = self.gzsl_map[key]['feat']
            if len(features) < n_samples:
                aug_features = [random.choice(features)
                                for _ in range(n_samples)]
            else:
                aug_features = random.sample(features, n_samples)
            label = self.gzsl_map[key]['label']
            dataset.extend([(torch.FloatTensor(f), label, key)
                           for f in aug_features])
        return dataset

    def create_orig_dataset(self):
        '''
            - Return dataset(feature, label_in_dataset, label_for_classification)
        '''
        if self.train:
            labels_index = self.trainval_index
            # classmap = self.train_classmap
            self.gzsl_map = dict()
        else:
            labels = self.att_splitst['test_unseen_loc'].reshape(-1)
            if self.gzsl:
                labels = np.concatenate(
                    (labels, self.att_splits['test_seen_loc'].reshape(-1)))
                classmap = {**self.train_classmap, **self.test_classmap}
            else:
                classmap = self.test_classmap

        dataset = []
        for li in labels_index:
            label = self.labels[li - 1]
            # the index of this label in the array(self.trainval_labels)
            label_idx = self.trainval_labels.tolist().index(label)
            print(label)
            print(label_idx)
            dataset.append((self.trainval_feats[li - 1], label))
            print(dataset[0])

            if self.train:
                # create a map bw class label and features
                if self.gzsl_map.get(label_idx, None):
                    try:
                        self.gzsl_map[label_idx]['feat'].append(
                            self.features[li - 1])
                    except Exception:
                        self.gzsl_map[label_idx]['feat'] = [
                            self.features[li - 1]]
                else:
                    self.gzsl_map[label_idx] = {}

                # Add the label to map
                self.gzsl_map[label_idx]['label'] = label
                print(self.gzsl_map)
                exit()

        return dataset

    def __getitem__(self, index):
        if self.synthetic:
            # choose an example from synthetic dataset
            img_feature, orig_label, label_idx = self.syn_dataset[index]
        else:
            # choose an example from original dataset
            img_feature, orig_label, label_idx = self.dataset[index]

        label_attr = self.attributes[orig_label - 1]
        return img_feature, label_attr, label_idx

    def __len__(self):
        if self.synthetic:
            return len(self.syn_dataset)
        else:
            return len(self.dataset)
