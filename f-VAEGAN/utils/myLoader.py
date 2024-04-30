
import torch
import numpy as np
import scipy.io as sio
from sklearn import preprocessing


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    
    return mapped_label


class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)

    def read_matdataset(self, opt):
        res101 = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + "res101.mat")
        feature = res101['features'].T
        label = res101['labels'].astype(int).squeeze() - 1

        att_splits = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + "att_splits.mat")
        trainval_loc = att_splits['trainval_loc'].squeeze() - 1
        # train_loc = att_splits['train_loc'].squeeze() - 1
        # val_unseen_loc = att_splits['val_loc'].squeeze() - 1
        test_seen_loc = att_splits['test_seen_loc'].squeeze() - 1
        test_unseen_loc = att_splits['test_unseen_loc'].squeeze() - 1    

        self.attribute = torch.from_numpy(att_splits['att'].T).float()
        self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),self.attribute.size(1))

        if opt.preprocessing:
            if opt.standardization:
                print('standardization...')
                scaler = preprocessing.StandardScaler()
            else:
                scaler = preprocessing.MinMaxScaler()
            
            _train_feature = scaler.fit_transform(feature[trainval_loc])
            _test_seen_feature = scaler.transform(feature[test_seen_loc])
            _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
            self.train_feature = torch.from_numpy(_train_feature).float()
            mx = self.train_feature.max()
            self.train_feature.mul_(1/mx)
            self.train_label = torch.from_numpy(label[trainval_loc]).long() 
            self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
            self.test_unseen_feature.mul_(1/mx)
            self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
            self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
            self.test_seen_feature.mul_(1/mx)
            self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            
        else:
            self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
            self.train_label = torch.from_numpy(label[trainval_loc]).long() 
            self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
            self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float() 
            self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)

        # original data + synthesized data
        self.train_feature_all = self.train_feature[:]
        self.train_label_all = self.train_label[:]

    def next_seen_batch(self, batch_size):
        """
            - random get a batch_size data
        """
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    def next_batch(self, batch_size):
        """
            - random get a batch_size data
        """
        ntrain_all = len(self.train_feature_all)
        idx = torch.randperm(ntrain_all)[0:batch_size]
        batch_feature = self.train_feature_all[idx]
        batch_label = self.train_label_all[idx]
        # batch_att = self.attribute[batch_label]
        return batch_feature, batch_label