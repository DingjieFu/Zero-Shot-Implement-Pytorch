#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File : SAE.py
# @Author : DingjieFu
# @Time : 2024/01/03 19:57:54
import scipy
import scipy.io
import argparse
import numpy as np


class SAE:
    def __init__(self, X, S, Lambda):
        """
            - X: d·N data matrix.
            - S: k·N semantic matrix.
            - Lambda: regularisation parameter.
        """
        self.x = X
        self.s = S
        self.ld = Lambda
    
    def calc_projection_matrix(self):
        A = np.dot(self.s, self.s.transpose())
        B = self.ld * np.dot(self.x, self.x.transpose())
        C = (1 + self.ld) * np.dot(self.s, self.x.transpose()) 
        W = scipy.linalg.solve_sylvester(A, B, C)
        return W


def normalizeFeature(X):
    """
        - X = d x N
        - d -> feature dimension, N -> the number of features
    """
    X = X + 1e-10 # for avoid RuntimeWarning: invalid value encountered in divide
    feature_norm = np.sum(X**2, axis = 1)**0.5 # l2-norm
    feature = X / feature_norm[:, np.newaxis]
    return feature


def distCosine(X, Y):
    """
        - X, Y -> matrics
        - X = n1 x d, Y = n2 x d
        - dist = n1 x n2
    """
    xx = np.sum(X**2, axis=1)**0.5
    X = X / xx[:, np.newaxis]
    yy = np.sum(Y**2, axis=1)**0.5
    Y = Y / yy[:, np.newaxis]
    dist = 1 - np.dot(X, Y.transpose())
    return dist


def zsl_acc(S_pred, S_gt, args):
    """
        - S_pred: predicted semantic labels -> d x N
        - S_gt: ground truth semantic labels -> d x N
        - d -> feature dimension, N -> the number of features
    """
    dist = 1 - distCosine(S_pred.transpose(), normalizeFeature(S_gt.transpose())) # (N x N)
    y_hit_k = np.zeros((dist.shape[0], args.HITK)) # (N x HITK) 在前k个结果中 这里只计算top-1

    for idx in range(0, dist.shape[0]):
        sorted_id = sorted(range(len(dist[idx,:])), key=lambda k: dist[idx,:][k], reverse=True)
        y_hit_k[idx,:] = args.test_labels[sorted_id[0:args.HITK]]

    n = 0
    for idx in range(0, dist.shape[0]):
        if args.test_labels[idx] in y_hit_k[idx,:]:
            n = n + 1
    zsl_accuracy = float(n) / dist.shape[0] * 100
    return zsl_accuracy, y_hit_k


if __name__ == '__main__':
    # ======================================== paramters ======================================== #
    parser = argparse.ArgumentParser(description='SAE')
    parser.add_argument('--dataset_path', type=str, default='dataset/',
						help='Name of the dataset')
    parser.add_argument('--dataset', type=str, default='AWA1',
						help='Name of the dataset')
    parser.add_argument('--ld', type=float, default=500000) # lambda
    args = parser.parse_args()
    # ======================================== data loader ======================================== #
    # dict_keys(['__header__', '__version__', '__globals__', 'image_files', 'features', 'labels'])
    res101 = scipy.io.loadmat(args.dataset_path + args.dataset + '/res101.mat')
    # dict_keys(['__header__', '__version__', '__globals__', 'allclasses_names', 'att', 
	#         'original_att', 'test_seen_loc', 'test_unseen_loc', 'train_loc', 'trainval_loc', 'val_loc'])
    att_splits = scipy.io.loadmat(args.dataset_path + args.dataset + '/att_splits.mat')
    trainval_loc = 'trainval_loc'
    test_loc = 'test_unseen_loc'
    Labels = res101['labels']
    Features = res101['features']
    Attributes = att_splits['att'] # (K x z)

    trainval_data = Features[:,np.squeeze(att_splits[trainval_loc]-1)] # (d x N)
    test_data = Features[:,np.squeeze(att_splits[test_loc]-1)]  # (d x N')

    labels_trainval = Labels[np.squeeze(att_splits[trainval_loc]-1)] # (N x 1)
    labels_test = Labels[np.squeeze(att_splits[test_loc]-1)]
    trainval_attr = Attributes[:, np.squeeze(labels_trainval-1)] # (K x N)
    test_attr = Attributes[:, np.squeeze(labels_test-1)] # (K x N')
    test_label_unique = np.unique(labels_test) 
    i = 0
    for label in test_label_unique:
        labels_test[labels_test == label] = i
        i = i + 1
    args.test_labels = np.squeeze(labels_test)
    
    # Normalize the data
    train_data = normalizeFeature(trainval_data)
    # ======================================== train ======================================== #
    model = SAE(train_data, trainval_attr, args.ld)
    W = model.calc_projection_matrix() # (k x d)
    # ======================================== test ======================================== #
    args.HITK = 1
    # [F --> S], projecting data from feature space to semantic space
    semantic_predicted = np.dot(test_data.transpose(), normalizeFeature(W)) # (N x K)
    zsl_accuracy, _ = zsl_acc(semantic_predicted.transpose(), test_attr, args)
    print(f'[1] zsl accuracy for {args.dataset} dataset [F >>> S]: {zsl_accuracy:.2f}%')
    # [S --> F], projecting from semantic to visual space
    test_predicted = np.dot(normalizeFeature(test_attr).transpose(), normalizeFeature(W)) # (N x d)
    zsl_accuracy, _ = zsl_acc(test_predicted.transpose(), test_data, args)
    print(f'[2] zsl accuracy for {args.dataset} dataset [S >>> F]: {zsl_accuracy:.2f}%')