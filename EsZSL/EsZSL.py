#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File : EsZSL.py
# @Author : DingjieFu
# @Time : 2024/01/02 18:59:40
import argparse
import scipy.io
import numpy as np
from sklearn.metrics import confusion_matrix


class EsZSL:
	def __init__(self, labels, features, signatures):
		"""
			- labels --> 标签
			- features --> 特征
			- signatures --> 属性
		"""
		# ======================================== 训练和测试时使用 ======================================== #
		self.labels_trainval = labels[np.squeeze(att_splits[trainval_loc]-1)] # shape -> (7057,1) | labels_val -> 训练集+验证集标签
		self.labels_test = labels[np.squeeze(att_splits[test_loc]-1)] # shape -> (2967,1) | labels_test -> 测试集标签
		self.trainval_labels_seen = np.unique(self.labels_trainval) # 100+50
		self.test_labels_unseen = np.unique(self.labels_test) # 50类
		i = 0 # 将labels_trainval中原来的标签值(1-200中的150个数)转换为trainval_labels_seen(150类)数组的索引(0-149)
		for label in self.trainval_labels_seen:
			self.labels_trainval[self.labels_trainval == label] = i
			i = i + 1
		j = 0 # 将labels_test中原来的标签值(1-200中的50个数)转换为test_labels_unseen(50类)数组的索引(0-49)
		for label in self.test_labels_unseen:
			self.labels_test[self.labels_test == label] = j
			j = j + 1
		self.trainval_vec = features[:,np.squeeze(att_splits[trainval_loc]-1)] # shape -> (2048,7057)
		self.test_vec = features[:,np.squeeze(att_splits[test_loc]-1)] # shape -> (2048,2967)
		self.trainval_sig = signatures[:,(self.trainval_labels_seen)-1] # (312,150)
		self.test_sig = signatures[:,(self.test_labels_unseen)-1] # (312,50)
		self.m_trainval = self.labels_trainval.shape[0] # 训练集+验证集实例数 7057
		self.z_trainval = len(self.trainval_labels_seen)	# 训练集+验证集种类数 150
	
		self.gt_trainval = 0*np.ones((self.m_trainval, self.z_trainval)) # Y
		self.gt_trainval[np.arange(self.m_trainval), np.squeeze(self.labels_trainval)] = 1

		# ======================================== 学习超参数时使用 ======================================== #
		self.labels_train = labels[np.squeeze(att_splits[train_loc]-1)] # shape -> (4702,1) | labels_train -> 训练集标签
		self.labels_val = labels[np.squeeze(att_splits[val_loc]-1)] # shape -> (2355,1) | labels_val -> 验证集标签
		self.train_labels_seen = np.unique(self.labels_train) # 100类
		self.val_labels_unseen = np.unique(self.labels_val) # 50类
		i = 0 # 将labels_train中原来的标签值(1-200中的100个数)转换为train_labels_seen(100类)数组的索引(0-99)
		for label in self.train_labels_seen:
			self.labels_train[self.labels_train == label] = i
			i = i + 1
		j = 0 # 将labels_val中原来的标签值(1-200中的50个数)转换为val_labels_unseen(50类)数组的索引(0-49)
		for label in self.val_labels_unseen:
			self.labels_val[self.labels_val == label] = j
			j = j + 1
		self.train_vec = features[:,np.squeeze(att_splits[train_loc]-1)] # shape -> (2048,4702)
		self.val_vec = features[:,np.squeeze(att_splits[val_loc]-1)] # shape -> (2048,2355)
		self.train_sig = signatures[:,(self.train_labels_seen)-1] # (312,100)
		self.val_sig = signatures[:,(self.val_labels_unseen)-1] # (312,50)
		self.m_train = self.labels_train.shape[0] # 训练集实例数 4702
		self.z_train = len(self.train_labels_seen) # 训练集种类数 100
		self.gt_train = 0*np.ones((self.m_train, self.z_train)) # Y
		self.gt_train[np.arange(self.m_train), np.squeeze(self.labels_train)] = 1

	def train(self, Gamma, Lambda):
		d_trainval = self.trainval_vec.shape[0] # 数据的维度d
		a_trainval = self.trainval_sig.shape[0] # 属性的维度a
		V = np.zeros((d_trainval,a_trainval)) # V(d,a)
		# self.trainval_vec(d,m) @ self.trainval_vec.transpose()(m,d) = (d,d) ==> X @ X.T
		# (10**Gamma)*np.eye(d_trainval)表示正则化项 其中np.eye(d_trainval)生成(d,d)的单位矩阵 ==> γI
		# np.linalg.pinv(A）求伪逆矩阵 part_1_test(d,d) ==> np.linalg.pinv(X @ X.T + γI)
		part_1_test = np.linalg.pinv(np.matmul(self.trainval_vec, self.trainval_vec.transpose()) + (10**Gamma)*np.eye(d_trainval))
		# self.trainval_vec(d,m) @ self.gt_trainval(m,z) = (d,z)  ==> X @ Y
		# (d,z) @ (z,a) = (d,a)  ==> X @ Y @ S.T
		part_0_test = np.matmul(np.matmul(self.trainval_vec,self.gt_trainval),self.trainval_sig.transpose())
		# self.trainval_sig(z,a) @ self.trainval_sig.transpose()(a,z) = (z,z)  ==> S @ S.T
		# (10**Lambda)*np.eye(a_trainval) ==> λI
		# part_2_test(a,a)  ==> np.linalg.pinv(S @ S.T + λI)
		part_2_test = np.linalg.pinv(np.matmul(self.trainval_sig, self.trainval_sig.transpose()) + (10**Lambda)*np.eye(a_trainval))
		# V = (d,d) @ (d,a) @ (a,a) = (d,a)
		# V = np.linalg.pinv(X @ X.T + γI) @ (X @ Y @ S.T) @ np.linalg.pinv(S @ S.T + λI) 论文中的表达式(5)
		V = np.matmul(np.matmul(part_1_test,part_0_test),part_2_test)
		return V

	def test(self,weights):
		# X.T @ V @ S
		# (m,z)(2967,50)
		outputs_1 = np.matmul(np.matmul(self.test_vec.transpose(),weights),self.test_sig)
		# argmax(x.T @ V @ Si)
		preds_1 = np.array([np.argmax(output) for output in outputs_1])
		cm = confusion_matrix(self.labels_test, preds_1)
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		acc = sum(cm.diagonal())/len(self.test_labels_unseen)
		print("The top-1 accuracy is:", acc)

	def find_hyperparams(self):
		"""
			- 在验证集上的最佳参数未必在测试集上也最优
		"""
		d_train = self.train_vec.shape[0]
		a_train = self.train_sig.shape[0]

		accu = 0.10
		Gamma0 = 4
		Lambda0 = 4
		# Weights
		V = np.zeros((d_train,a_train))
		for Gamma in range(-3, 4):
			for Lambda in range(-3,4):
				#One line solution
				part_1 = np.linalg.pinv(np.matmul(self.train_vec, self.train_vec.transpose()) + (10**Gamma)*np.eye(d_train))
				part_0 = np.matmul(np.matmul(self.train_vec,self.gt_train),self.train_sig.transpose())
				part_2 = np.linalg.pinv(np.matmul(self.train_sig, self.train_sig.transpose()) + (10**Lambda)*np.eye(a_train))		
				V = np.matmul(np.matmul(part_1,part_0),part_2)	
				#predictions
				outputs = np.matmul(np.matmul(self.val_vec.transpose(),V),self.val_sig)
				preds = np.array([np.argmax(output) for output in outputs])		
				cm = confusion_matrix(self.labels_val, preds)
				cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
				Acc = sum(cm.diagonal())/len(self.val_labels_unseen)
				if Acc > accu:
					accu = Acc
					Gamma0 = Gamma
					Lambda0 = Lambda
				print(f"Gamma={Gamma}, Lambda={Lambda}, Acc={Acc}")
		print(f"\nBest params: Gamma={Gamma0}, Lambda={Lambda0} | Acc={accu}")
		return Gamma0, Lambda0


if __name__ == "__main__":
	# ======================================== paramters ======================================== #
	parser = argparse.ArgumentParser(description='EsZSL')
	parser.add_argument('--dataset_path', type=str, default='dataset/',
						help='Name of the dataset')
	parser.add_argument('--dataset', type=str, default='CUB',
						help='Name of the dataset')
	parser.add_argument('--Gamma', type=int, default=0,
						help='value of hyper-parameter')
	parser.add_argument('--Lambda', type=int, default=0,
						help='value of hyper-parameter')
	args = parser.parse_args()
	# ======================================== data loader ======================================== #
	# 字典 dict_keys(['__header__', '__version__', '__globals__', 'image_files', 'features', 'labels'])
	res101 = scipy.io.loadmat(args.dataset_path + args.dataset + '/res101.mat')
	# 字典 dict_keys(['__header__', '__version__', '__globals__', 'allclasses_names', 'att', 
	#         'original_att', 'test_seen_loc', 'test_unseen_loc', 'train_loc', 'trainval_loc', 'val_loc'])
	att_splits = scipy.io.loadmat(args.dataset_path + args.dataset + '/att_splits.mat')

	train_loc = 'train_loc' # resNet101.mat中训练集特征的实例索引
	val_loc = 'val_loc' # resNet101.mat中验证集特征的实例索引
	trainval_loc = 'trainval_loc' # resNet101.mat中训练集+验证集 特征的实例索引
	test_loc = 'test_unseen_loc'#  # resNet101.mat中测试集特征的实例索引
	Labels = res101['labels'] # (11788,1) 11788个样本 label为 1-200 共200个标签
	X_features = res101['features'] # shape --> (2048, 11788) 其列对应与实例
	Signatures = att_splits['att'] # shape --> (312,200) 列对应于标准化为具有单位L2范数的类属性向量
	# ======================================== model ======================================== #
	model = EsZSL(Labels, X_features, Signatures)
	Gamma, Lambda = args.Gamma, args.Lambda
	if not Gamma and not Lambda:
		Gamma, Lambda = model.find_hyperparams()
	weights = model.train(Gamma, Lambda)
	model.test(weights)