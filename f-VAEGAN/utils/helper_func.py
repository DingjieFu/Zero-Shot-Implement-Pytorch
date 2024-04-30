import torch
import numpy as np


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size()).fill_(-1)
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label


def val_gzsl(test_X, test_label, target_classes, in_package):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]  # number of samples
        predicted_label = torch.LongTensor(test_label.size())
        for _ in range(0, ntest, batch_size):
            end = min(ntest, start+batch_size)
            input = test_X[start:end].to(device)
            output= model(input)
            predicted_label[start:end] = torch.argmax(output.data, 1)
            start = end
        acc = compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package)
        return acc


def val_zs_gzsl(test_X, test_label, unseen_classes, in_package):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        predicted_label_gzsl = torch.LongTensor(test_label.size())
        predicted_label_zsl = torch.LongTensor(test_label.size())
        predicted_label_zsl_t = torch.LongTensor(test_label.size())
        for i in range(0, ntest, batch_size):
            end = min(ntest, start+batch_size)
            input = test_X[start:end].to(device)
            output = model(input)
            output_t = output.clone()
            output_t[:, unseen_classes] = output_t[:,unseen_classes]+torch.max(output)+1
            predicted_label_zsl[start:end] = torch.argmax(output_t.data, 1)
            predicted_label_zsl_t[start:end] = torch.argmax(
                output.data[:, unseen_classes], 1)
            predicted_label_gzsl[start:end] = torch.argmax(output.data, 1)
            start = end
        acc_gzsl = compute_per_class_acc_gzsl(
            test_label, predicted_label_gzsl, unseen_classes, in_package)
        # acc_zs = compute_per_class_acc_gzsl(test_label, predicted_label_zsl, unseen_classes, in_package)
        acc_zs_t = compute_per_class_acc(map_label(
            test_label, unseen_classes), predicted_label_zsl_t, unseen_classes.size(0))
        # assert np.abs(acc_zs - acc_zs_t) < 0.001
        # print('acc_zs: {} acc_zs_t: {}'.format(acc_zs,acc_zs_t))
        return acc_gzsl, acc_zs_t


def compute_per_class_acc(test_label, predicted_label, nclass):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)
        acc_per_class[i] = torch.sum(
            test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
    return acc_per_class.mean().item()


def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package):
    device = in_package['device']
    per_class_accuracies = torch.zeros(
        target_classes.size()[0]).float().to(device).detach()
    predicted_label = predicted_label.to(device)
    for i in range(target_classes.size()[0]):
        is_class = test_label == target_classes[i]
        per_class_accuracies[i] = torch.div(
            (predicted_label[is_class] == test_label[is_class]).sum().float(), is_class.sum().float())
    return per_class_accuracies.mean().item()


def eval_zs_gzsl(dataloader, model, device):
    model.eval()
    test_seen_feature = dataloader.test_seen_feature
    test_seen_label = dataloader.test_seen_label.to(device)

    test_unseen_feature = dataloader.test_unseen_feature
    test_unseen_label = dataloader.test_unseen_label.to(device)
    seenclasses = dataloader.seenclasses
    unseenclasses = dataloader.unseenclasses
    batch_size = 100
    in_package = {'model': model, 'device': device, 'batch_size': batch_size}

    with torch.no_grad():
        acc_seen = val_gzsl(test_seen_feature, test_seen_label, seenclasses, in_package)
        acc_novel, acc_zs = val_zs_gzsl(test_unseen_feature, test_unseen_label, unseenclasses, in_package)
    if (acc_seen+acc_novel) > 0:
        H = (2*acc_seen*acc_novel) / (acc_seen+acc_novel)
    else:
        H = 0
    return acc_seen, acc_novel, H, acc_zs


def eval_train_acc(dataloader, model, device):
    model.eval()
    test_feature = dataloader.train_feature_all
    test_label = dataloader.train_label_all.to(device)
    batch_size = 100
    with torch.no_grad():
        start = 0
        ntest = test_feature.size()[0]  # number of samples
        predicted_label = torch.LongTensor(test_label.size()).to(device)
        for _ in range(0, ntest, batch_size):
            end = min(ntest, start+batch_size)
            input = test_feature[start:end].to(device)
            output= model(input)
            predicted_label[start:end] = torch.argmax(output.data, 1)
            start = end
        acc = (predicted_label == test_label).sum().float() / len(test_label)
        return acc