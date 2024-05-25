

import random
import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import models.fVAEGAN as model
from utils.myLoader import MyLoader, map_label
import classifiers.classifier_images as classifier
from config import args




cudnn.benchmark = True
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
print("Random Seed: ", args.seed)

# load data
data = MyLoader(args)
print("# of training samples: ", data.ntrain)

netE = model.Encoder(args).to(args.device)
netG = model.Generator(args).to(args.device)
netD = model.Discriminator_D1(args).to(args.device)

print(netE)
print(netG)
print(netD)


###########
# Init Tensors
input_res = torch.FloatTensor(args.batch_size, args.resSize).to(args.device)
input_att = torch.FloatTensor(args.batch_size, args.attSize).to(args.device)
noise = torch.FloatTensor(args.batch_size, args.nz).to(args.device)

##########

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(), reduction='sum')
    BCE = BCE.sum()/ x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    return (BCE + KLD)
           
def sample():
    batch_feature, batch_att = data.next_seen_batch(args.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)

    
def generate_syn_feature(generator,classes, attribute,num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, args.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, args.attSize)
    syn_noise = torch.FloatTensor(num, args.nz)
    if args.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    with torch.no_grad():
        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            syn_att.copy_(iclass_att.repeat(num, 1))
            syn_noise.normal_(0, 1)
            syn_noisev = Variable(syn_noise)
            syn_attv = Variable(syn_att)
            fake = generator(syn_noisev,c=syn_attv)

            output = fake
            syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label


optimizer   = optim.Adam(netE.parameters(), lr=args.lr)
optimizerD  = optim.Adam(netD.parameters(), lr=args.lr,betas=(0.5, 0.999))
optimizerG  = optim.Adam(netG.parameters(), lr=args.lr,betas=(0.5, 0.999))


def calc_gradient_penalty(netD,real_data, fake_data, input_att):
    alpha = torch.rand(args.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if args.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if args.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    if args.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.lambda1
    return gradient_penalty

best_gzsl_acc = 0
best_zsl_acc = 0

for epoch in range(0,args.nepoch):
    for _ in range(0, args.loop):
        for i in range(0, data.ntrain, args.batch_size):
            ######### Discriminator training ##############
            for p in netD.parameters(): # unfreeze discrimator
                p.requires_grad = True

            # Train D1 and Decoder (and Decoder Discriminator)
            gp_sum = 0 # lAMBDA VARIABLE
            for iter_d in range(args.critic_iter):
                sample()
                netD.zero_grad()          
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)

                criticD_real = netD(input_resv, input_attv)
                criticD_real = -args.gammaD * criticD_real.mean()
                criticD_real.backward()
                if args.encoded_noise:        
                    means, log_var = netE(input_resv, input_attv)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([args.batch_size, args.latent_size]).cpu()
                    eps = Variable(eps.cuda())
                    z = eps * std + means
                else:
                    noise.normal_(0, 1)
                    z = Variable(noise)

                fake = netG(z, c=input_attv)

                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake = args.gammaD*criticD_fake.mean()
                criticD_fake.backward()
                # gradient penalty
                gradient_penalty = args.gammaD*calc_gradient_penalty(netD, input_res, fake.data, input_att)
                # if args.lambda_mult == 1.1:
                gp_sum += gradient_penalty.data
                gradient_penalty.backward()         
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty #add Y here and #add vae reconstruction loss
                optimizerD.step()

            gp_sum /= (args.gammaD*args.lambda1*args.critic_iter)
            if (gp_sum > 1.05).sum() > 0:
                args.lambda1 *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                args.lambda1 /= 1.1

            #############Generator training ##############
            # Train Generator and Decoder
            for p in netD.parameters(): #freeze discrimator
                p.requires_grad = False

            netE.zero_grad()
            netG.zero_grad()

            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            means, log_var = netE(input_resv, input_attv)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([args.batch_size, args.latent_size]).cpu()
            eps = Variable(eps.cuda())
            z = eps * std + means

            recon_x = netG(z, c=input_attv)

            vae_loss_seen = loss_fn(recon_x, input_resv, means, log_var)
            errG = vae_loss_seen
            
            if args.encoded_noise:
                criticG_fake = netD(recon_x, input_attv).mean()
                fake = recon_x 
            else:
                noise.normal_(0, 1)
                noisev = Variable(noise)

                fake = netG(noisev, c=input_attv)
                criticG_fake = netD(fake,input_attv).mean()
                
            G_cost = -criticG_fake
            errG += args.gammaG * G_cost
            errG.backward()

            optimizer.step()
            optimizerG.step()

        
    print('[%d/%d]  Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f'% 
          (epoch, args.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(),vae_loss_seen.item()),end=" ")
    netG.eval()
    syn_feature, syn_label = generate_syn_feature(netG,data.unseenclasses, data.attribute, args.syn_num)
    # Generalized zero-shot learning
  
    # Concatenate real seen features with synthesized unseen features
    train_X = torch.cat((data.train_feature, syn_feature), 0)
    train_Y = torch.cat((data.train_label, syn_label), 0)
    nclass = args.nclass_all
    # Train GZSL classifier
    gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, data, nclass, args.cuda, args.classifier_lr, 0.5, \
            25, args.syn_num, generalized=True, netDec=None, dec_size=args.attSize, dec_hidden_size=4096)
    if best_gzsl_acc < gzsl_cls.H:
        best_acc_seen, best_acc_unseen, best_gzsl_acc = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H
    print('GZSL: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H),end=" ")

    # Zero-shot learning
    # Train ZSL classifier
    zsl_cls = classifier.CLASSIFIER(syn_feature, map_label(syn_label, data.unseenclasses), \
                    data, data.unseenclasses.size(0), args.cuda, args.classifier_lr, 0.5, 25, args.syn_num, \
                    generalized=False, netDec=None, dec_size=args.attSize, dec_hidden_size=4096)
    acc = zsl_cls.acc
    if best_zsl_acc < acc:
        best_zsl_acc = acc
    print('ZSL: unseen accuracy=%.4f' % (acc))
    netG.train()


print('Dataset', args.dataset)
print('the best ZSL unseen accuracy is', best_zsl_acc)
print('the best GZSL seen accuracy is', best_acc_seen)
print('the best GZSL unseen accuracy is', best_acc_unseen)
print('the best GZSL H is', best_gzsl_acc)
