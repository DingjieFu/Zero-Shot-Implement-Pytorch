import torch
from torch.utils.data import DataLoader

import os
import scipy.io
import argparse
import numpy as np
from trainer import Trainer
from myData import ZSLDataset


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getDatasetParam(dset):
    """
        - get parameters from dataset
    """
    res101 = scipy.io.loadmat('%s/res101.mat' % dset)  # feature dict
    att_splits = scipy.io.loadmat('%s/att_splits.mat' % dset)  # attribute dict
    Labels = res101['labels']  # (sample_nums x 1)
    X_features = res101['features']  # (feature_dims x sample_nums)
    S_features = att_splits['att']  # (attribute_dims x class_nums)
    # (trainval_samples x 1)
    labels_trainval = Labels[np.squeeze(att_splits['trainval_loc']-1)]
    # (test_samples x 1)
    labels_test = Labels[np.squeeze(att_splits['test_unseen_loc']-1)]
    # retrun params
    x_dim = X_features.shape[0]
    z_dim = S_features.shape[0]
    attr_dim = S_features.shape[0]
    n_train = np.unique(labels_trainval).shape[0]
    n_test = np.unique(labels_test).shape[0]
    return x_dim, z_dim, attr_dim, n_train, n_test


if __name__ == "__main__":
    # ========================================parameters======================================== #
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='dataset/',
                        help='Name of the dataset')
    parser.add_argument('--dataset', type=str, default='CUB',
                        help='Name of the dataset')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--HITK', type=int,
                        default=1, help='Hit k criterion')
    parser.add_argument('--lmbda', type=float, default=10.0,
                        help='Penalty coefficient')
    parser.add_argument('--beta', type=float, default=0.01,
                        help="Hyperparameter weighting the classifier")
    parser.add_argument('--gzsl', action='store_true',
                        default=False, help='Boolean for Generalized ZSL')
    parser.add_argument('--use_cls_loss', action='store_true',
                        default=False, help='Boolean for cls_loss')

    args = parser.parse_args()

    # ========================================DATALOADER======================================== #
    x_dim, z_dim, attr_dim, n_train, n_test = getDatasetParam(
        args.dataset_path + args.dataset)

    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 0,
        'drop_last': True
    }

    train_dataset = ZSLDataset(
        args.dataset_path + args.dataset, n_train, n_test, args.gzsl)
    train_loader = DataLoader(train_dataset, **params)

    # trainer object for mini batch training
    train_agent = Trainer(
        device, x_dim, z_dim, attr_dim,
        n_train=n_train, n_test=n_test, gzsl=args.gzsl,
        n_critic=args.n_critic, lmbda=args.lmbda, beta=args.beta,
        batch_size=args.batch_size
    )
    # ===================================PRETRAIN THE SOFTMAX CLASSIFIER=================================== #
    model_name = "%s_classifier" % args.dataset
    success = train_agent.load_model(model=model_name)
    if success:
        print("Discriminative classifier parameters loaded...")
    else:
        print("Training the discriminative classifier...")
        for ep in range(1, args.n_epochs + 1):
            loss = 0
            for idx, (img_features, label_attr, label_idx) in enumerate(train_loader):
                l = train_agent.fit_classifier(
                    img_features, label_attr, label_idx)
                loss += l

            print("Loss for epoch: %3d - %.4f" % (ep, loss))
        train_agent.save_model(model=model_name)

    # ========================================TRAIN THE GAN======================================== #
    model_name = "%s_gan" % args.dataset
    success = train_agent.load_model(model=model_name)
    if success:
        print("\nGAN parameters loaded....")
    else:
        print("\nTraining the GAN...")
        for ep in range(1, args.n_epochs + 1):
            loss_dis = 0
            loss_gan = 0
            for idx, (img_features, label_attr, label_idx) in enumerate(train_loader):
                l_d, l_g = train_agent.fit_GAN(
                    img_features, label_attr, label_idx, args.use_cls_loss)
                loss_dis += l_d
                loss_gan += l_g
            print("Loss for epoch: %3d - D: %.4f | G: %.4f"
                  % (ep, loss_dis, loss_gan))

        train_agent.save_model(model=model_name)

    # ==============================TRAIN FINAL CLASSIFIER ON SYNTHETIC DATASET============================== #
    # create new synthetic dataset using trained Generator
    seen_dataset = None
    if args.gzsl:
        seen_dataset = train_dataset.gzsl_dataset

    syn_dataset = train_agent.create_syn_dataset(
        train_dataset.test_classmap, train_dataset.attributes, seen_dataset)
    final_dataset = ZSLDataset(args.dataset_path + args.dataset, n_train, n_test,
                               gzsl=args.gzsl, is_train=True, synthetic=True, syn_dataset=syn_dataset)
    final_train_generator = DataLoader(final_dataset, **params)

    model_name = "%s_final_classifier" % args.dataset
    success = train_agent.load_model(model=model_name)
    if success:
        print("\nFinal classifier parameters loaded....")
    else:
        print("\nTraining the final classifier on the synthetic dataset...")
        for ep in range(1, args.n_epochs + 1):
            syn_loss = 0
            for idx, (img, label_attr, label_idx) in enumerate(final_train_generator):
                l = train_agent.fit_final_classifier(
                    img, label_attr, label_idx)
                syn_loss += l
            # print losses on real and synthetic datasets
            print("Loss for epoch: %3d - %.4f" % (ep, syn_loss))
        train_agent.save_model(model=model_name)

    # ========================================TESTING PHASE======================================== #
    test_dataset = ZSLDataset(args.dataset_path + args.dataset, n_train,
                              n_test, gzsl=args.gzsl, is_train=False)
    test_loader = DataLoader(test_dataset, **params)

    print("\nFinal Accuracy on ZSL Task: %.3f" % train_agent.test(test_loader))
