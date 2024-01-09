import torch
from torch.utils.data import DataLoader

import os
import argparse
from myData import ZSLDataset
from trainer import Trainer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"


def getDatasetParam(dset):
    """
        - get parameters from dataset
    """
    pass


if __name__ == "__main__":
    # ========================================parameters======================================== #
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='dataset/',
                        help='Name of the dataset')
    parser.add_argument('--dataset', type=str, default='CUB',
                        help='Name of the dataset')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--n_critic', type=int,
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

    if args.dataset == 'APY':
        x_dim = 2048
        attr_dim = 64
        n_train = 20
        n_test = 12
    elif args.dataset == 'AWA1' or args.dataset == 'AWA2':
        x_dim = 2048
        attr_dim = 85
        n_train = 40
        n_test = 10
    elif args.dataset == 'CUB':
        x_dim = 2048
        z_dim = 312
        attr_dim = 312
        n_train = 150
        n_test = 50
    elif args.dataset == 'SUN':
        x_dim = 2048
        attr_dim = 102
        n_train = 645
        n_test = 72
    else:
        raise NotImplementedError

    n_epochs = args.n_epochs

    # trainer object for mini batch training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_agent = Trainer(
        device, x_dim, z_dim, attr_dim,
        n_train=n_train, n_test=n_test, gzsl=args.gzsl,
        n_critic=args.n_critic, lmbda=args.lmbda, beta=args.beta,
        batch_size=args.batch_size
    )

    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 0,
        'drop_last': True
    }

    train_dataset = ZSLDataset(args.dataset, n_train, n_test, args.gzsl)
    train_loader = DataLoader(train_dataset, **params)

    # ===================================PRETRAIN THE SOFTMAX CLASSIFIER=================================== #
    model_name = "%s_classifier" % args.dataset
    success = train_agent.load_model(model=model_name)
    if success:
        print("Discriminative classifier parameters loaded...")
    else:
        print("Training the discriminative classifier...")
        for ep in range(1, n_epochs + 1):
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
        for ep in range(1, n_epochs + 1):
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
    final_dataset = ZSLDataset(args.dataset, n_train, n_test,
                               gzsl=args.gzsl, train=True, synthetic=True, syn_dataset=syn_dataset)
    final_train_generator = DataLoader(final_dataset, **params)

    model_name = "%s_final_classifier" % args.dataset
    success = train_agent.load_model(model=model_name)
    if success:
        print("\nFinal classifier parameters loaded....")
    else:
        print("\nTraining the final classifier on the synthetic dataset...")
        for ep in range(1, n_epochs + 1):
            syn_loss = 0
            for idx, (img, label_attr, label_idx) in enumerate(final_train_generator):
                l = train_agent.fit_final_classifier(
                    img, label_attr, label_idx)
                syn_loss += l
            # print losses on real and synthetic datasets
            print("Loss for epoch: %3d - %.4f" % (ep, syn_loss))
        train_agent.save_model(model=model_name)

    # ========================================TESTING PHASE======================================== #
    test_dataset = ZSLDataset(args.dataset, n_train,
                              n_test, gzsl=args.gzsl, train=False)
    test_loader = DataLoader(test_dataset, **params)

    print("\nFinal Accuracy on ZSL Task: %.3f" % train_agent.test(test_loader))
