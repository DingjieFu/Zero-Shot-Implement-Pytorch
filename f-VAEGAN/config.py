import argparse


parser = argparse.ArgumentParser()


parser.add_argument('--seed', type=int, default=2024, help='Seed for reproduction')
parser.add_argument('--device', default='cuda:0', help='cpu/cuda:x')
# ======================================== Path Config ======================================== #
parser.add_argument('--matRoot', default='data', help='path to xlsa17')

# ======================================== Data Config ======================================== #
parser.add_argument('--dataset', default='AWA2', help='[AWA2,CUB, SUN]')


# ======================================== Training Config ======================================== #
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')


# ======================================== VAEGAN Config ======================================== #
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lambda2', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate to train GANs ')
parser.add_argument('--gammaD', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--gammaG', type=int, default=1000, help='weight on the W-GAN loss')





parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)

parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--encoded_noise', action='store_true', default=False, help='enables validation mode')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--validation', action='store_true', default=False, help='enables validation mode')
parser.add_argument("--encoder_layer_sizes", type=list, default=[8192, 4096])
parser.add_argument("--decoder_layer_sizes", type=list, default=[4096, 8192])

parser.add_argument("--latent_size", type=int, default=312)
parser.add_argument('--loop', type=int, default=2)


args = parser.parse_args()
args.lambda2 = args.lambda1
args.encoder_layer_sizes[0] = args.resSize
args.decoder_layer_sizes[-1] = args.resSize
args.latent_size = args.attSize
