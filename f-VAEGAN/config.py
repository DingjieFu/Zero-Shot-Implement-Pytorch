"""
    - Attention to set root paths
"""
import os
import argparse


projectPath = os.path.dirname(__file__)
parser = argparse.ArgumentParser()
# -------------------- Path config -------------------- #
parser.add_argument('--dataset', type=str, default='CUB', help='[AWA2,CUB,SUN]')
parser.add_argument('--dataroot', type=str, default=projectPath+'/data', help='path to dataset')
parser.add_argument('--result_root', type=str, default=projectPath+'/result', help='root path for saving checkpoint')
parser.add_argument('--save', action='store_true', default=False, help='determine to save model weights')
# -------------------- device config -------------------- #
parser.add_argument('--device', default='cuda:0', help='cpu/cuda:x')

# -------------------- training config -------------------- #
parser.add_argument('--backbone', default='Resnet', help='VIT/Resnet')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--clf_nepoch', type=int, default=20, help='number of epochs to train softmax clf')
parser.add_argument('--early_stop', type=int, default=10, help='early break')
# -------------------- model config -------------------- #
parser.add_argument("--latent_size", type=int, default=312)
parser.add_argument('--z_dim', type=int, default=312, help='dim of the latent z vector')
parser.add_argument('--attSize', type=int, default=312, help='size of semantic features')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')

# -------------------- f-VAEGAN config -------------------- #
parser.add_argument('--loop', type=int, default=2)
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--encoded_noise', action='store_true', default=False, help='enables validation mode')
parser.add_argument('--gammaD', type=int, default=10, help='weight on the W-GAN loss')
parser.add_argument('--gammaG', type=int, default=10, help='weight on the W-GAN loss')

parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lambda2', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')

parser.add_argument('--syn_num', type=int, default=500, help='number features to generate per class')
parser.add_argument('--preprocessing', action='store_true', default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)

parser.add_argument('--lr', type=float, default=0.001, help='learning rate to train GANs')
parser.add_argument('--feed_lr', type=float, default=0.0001, help='learning rate to train GANs')
parser.add_argument('--dec_lr', type=float, default=0.0001, help='learning rate to train GANs')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--manualSeed', type=int, default=2024, help='seed for reproduction')

opt = parser.parse_args()
