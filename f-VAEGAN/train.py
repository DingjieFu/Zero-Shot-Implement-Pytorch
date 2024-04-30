
import os
import time
import torch
import random
import logging
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torch.backends.cudnn as cudnn


from config import opt
import utils.myLoader as myLoader
from model.myModel import Encoder, Generator, Discriminator, Myclassifier
from utils.helper_func import eval_zs_gzsl

# ======================================== Global setting ======================================== #
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
device = opt.device


def create_unique_folder_name(base_folder_path):
    count = 0
    new_folder_name = base_folder_path
    while os.path.exists(new_folder_name):
        count += 1
        new_folder_name = f"{base_folder_path}({count})"
    return new_folder_name


def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x + 1e-12, x, reduction='sum')
    BCE = BCE.sum()/ x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    return (BCE + KLD)


def generate_syn_feature(generator, classes, attribute, num, device):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize).to(device)
    syn_noise = torch.FloatTensor(num, opt.z_dim).to(device)

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        with torch.no_grad():
            syn_noisev = syn_noise.normal_(0, 1)
            syn_attv = syn_att.copy_(iclass_att.repeat(num, 1))
        fake = generator(syn_noisev,c=syn_attv)
        output = fake
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)
    return syn_feature, syn_label


def calc_gradient_penalty(netD, real_data, fake_data, input_att, device):
    alpha = torch.rand(real_data.size(0), 1, device=device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data).to(device).requires_grad_(True)
    disc_interpolates = netD(interpolates, input_att)
    grad_outputs = torch.ones_like(disc_interpolates, device=device)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=grad_outputs,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


dataloader = myLoader.DATA_LOADER(opt)
print(" ==> Number of training samples: ", dataloader.ntrain)


if opt.backbone == "Resnet":
    feature_dim = 2048
else:
    raise ValueError("Unkonwn backbone!")
if opt.dataset == "AWA2":
    att_dim = 85
elif opt.dataset == "CUB":
    att_dim = 312
elif opt.dataset == "SUN":
    att_dim = 102
else:
    raise ValueError("Unknown dataset!")
latent_dim = z_dim = att_dim


netE = Encoder(feature_dim, att_dim, latent_dim).to(device)
netG = Generator(z_dim, att_dim, feature_dim).to(device)
netD = Discriminator(att_dim).to(device)
netCLF = Myclassifier(feature_dim, len(dataloader.allclasses)).to(device)
# print(netE, "\n", netG, "\n", netD, "\n", netclf)
optimizer  = optim.Adam(netE.parameters(), lr=opt.lr)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizerCLF = optim.Adam(netCLF.parameters(), lr=opt.classifier_lr, betas=(0.5, 0.999))
criterion = torch.nn.NLLLoss().to(device)

# ---------- run log ----------
os.makedirs(opt.result_root, exist_ok=True)
outlogDir = "{}/{}".format(opt.result_root, opt.dataset)
os.makedirs(outlogDir, exist_ok=True)
num_exps = len([f.path for f in os.scandir(outlogDir) if f.is_dir()])
outlogPath = os.path.join(outlogDir, create_unique_folder_name(outlogDir + f"/exp{num_exps}"))
os.makedirs(outlogPath, exist_ok=True)
t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
log = outlogPath + "/" + t + '.txt'
logging.basicConfig(format='%(message)s', level=logging.INFO,
                    filename=log,
                    filemode='w')
logger = logging.getLogger(__name__)
argsDict = opt.__dict__
for eachArg, value in argsDict.items():
    logger.info(eachArg + ':' + str(value))
logger.info("="*50)
    

best_record = [0.0, 0.0, 0.0, 0.0]
early_stop = 0
for epoch in range(0, opt.nepoch):
    for loop in range(0, opt.loop):
        for i in range(0, dataloader.ntrain, opt.batch_size):
            # ======================================== Discriminator training ======================================== #
            """
                - expect D to classify real-ones while discerning synthesized-ones
            """
            for p in netD.parameters(): # unfreeze discrimator
                p.requires_grad = True

            # Train D1 and Decoder (and Decoder Discriminator)
            gp_sum = 0 # lAMBDA VARIABLE
            for iter_d in range(opt.critic_iter):
                # sample()
                input_feature, _, input_att = dataloader.next_seen_batch(opt.batch_size)
                input_feature, input_att = input_feature.to(device), input_att.to(device)
                netD.zero_grad()
                
                if opt.encoded_noise: 
                    means, log_var = netE(input_feature, input_att)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([opt.batch_size, opt.latent_size]).to(device)
                    z = eps * std + means # torch.Size([bs, 312])
                else:
                    z = torch.FloatTensor(opt.batch_size, z_dim).normal_(0, 1).to(device)

                criticD_real = netD(input_feature, input_att)
                criticD_real = opt.gammaD * criticD_real.mean()
                # criticD_real.backward(mone)

                fake = netG(z, c=input_att)
                criticD_fake = netD(fake, input_att)
                criticD_fake = opt.gammaD * criticD_fake.mean()
                loss = criticD_fake - criticD_real
                # criticD_fake.backward(one)
                loss.backward()
                # gradient penalty
                gradient_penalty = opt.gammaD * calc_gradient_penalty(netD, input_feature, fake.data, input_att, device)
                gp_sum += gradient_penalty.data
                gradient_penalty.backward()         
                Wasserstein_D = criticD_real - criticD_fake
                D_loss = criticD_fake - criticD_real + gradient_penalty
                optimizerD.step()
    
            gp_sum /= (opt.gammaD * opt.lambda1 * opt.critic_iter)
            if (gp_sum > 1.05).sum() > 0:
                opt.lambda1 *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                opt.lambda1 /= 1.1

            # ======================================== Generator training ======================================== #
            # Train Generator and Decoder
            for p in netD.parameters():
                p.requires_grad = False

            netE.zero_grad()
            netG.zero_grad()

            # Decoder
            means, log_var = netE(input_feature, input_att)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([opt.batch_size, opt.latent_size]).to(device)
            z = eps * std + means # torch.Size([bs, 312])

            recon_x = netG(z, c=input_att)
            vae_loss_seen = loss_fn(recon_x, input_feature, means, log_var)
            errG = vae_loss_seen

            if opt.encoded_noise:
                criticG_fake = netD(recon_x, input_att).mean()
            else:
                noisev = torch.FloatTensor(opt.batch_size, z_dim).normal_(0, 1).to(device)
                fake = netG(noisev, c = input_att)
                criticG_fake = netD(fake, input_att).mean()

            G_loss = -criticG_fake
            errG += opt.gammaG * G_loss
            errG.backward()
            optimizer.step()
            optimizerG.step()

    print('[%d/%d]  Loss_D: %.4f, Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f' % \
          (epoch, opt.nepoch, D_loss.item(), G_loss.item(), Wasserstein_D.item(),vae_loss_seen.item()))
    logger.info('[%d/%d]  Loss_D: %.4f, Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f' % \
          (epoch, opt.nepoch, D_loss.item(), G_loss.item(), Wasserstein_D.item(),vae_loss_seen.item()))

    netG.eval()
    syn_feature, syn_label = generate_syn_feature(netG, dataloader.unseenclasses, dataloader.attribute, opt.syn_num, device)

    # Concatenate real seen features with synthesized unseen features
    dataloader.train_feature_all = torch.cat((dataloader.train_feature, syn_feature), 0)
    dataloader.train_label_all = torch.cat((dataloader.train_label, syn_label), 0)

    # ======================================== Train GZSL classifier ======================================== #
    best_record_miniEpoch = [0.0, 0.0, 0.0, 0.0]
    for _ in range(opt.clf_nepoch):
        for _ in range(0, len(dataloader.train_feature_all), opt.batch_size):      
            netCLF.zero_grad()
            batch_input, batch_label = dataloader.next_batch(opt.batch_size) 
            batch_input, batch_label = batch_input.to(device), batch_label.to(device)
            output = netCLF(batch_input)
            loss = criterion(output, batch_label)
            loss.backward()
            optimizerCLF.step()
        # acc_train = eval_train_acc(dataloader, netCLF, device)
        S, U, H, CZSL = eval_zs_gzsl(dataloader, netCLF, device)
        if CZSL > best_record[3]:
            best_record_miniEpoch[3] = CZSL
        if H > best_record_miniEpoch[2]:
            best_record_miniEpoch[:3] = [S, U, H]

    if best_record_miniEpoch[3] > best_record[3]:
        best_record[3] = best_record_miniEpoch[3]
    if best_record_miniEpoch[2] > best_record[2]:
        best_record[:3] = best_record_miniEpoch[:3]
        if opt.save:
            torch.save(netG.state_dict(), f"{outlogPath}/netG.pth")
            torch.save(netCLF.state_dict(), f"{outlogPath}/netCLF.pth")
            print('Models saved!')
        early_stop = 0
    early_stop += 1

    dictNow = {'Epoch': epoch, 'S': best_record_miniEpoch[0], 'U': best_record_miniEpoch[1], 'H': best_record_miniEpoch[2], 'CZSL': best_record_miniEpoch[3]}
    dictBest = {'S': best_record[0], 'U': best_record[1], 'H': best_record[2], 'CZSL': best_record[3]}
    print(f'Performance => S: {dictNow["S"]}, U: {dictNow["U"]}, H: {dictNow["H"]}, CZSL: {dictNow["CZSL"]}')
    print(f'Best GZSL|CZSL => S: {dictBest["S"]}, U: {dictBest["U"]}, H: {dictBest["H"]}, CZSL: {dictBest["CZSL"]}')

    logger.info("Performance => S:{:.6f}; U:{:.6f}; H:{:.6f}; CZSL:{:.6f}".format(
            dictNow['S'], dictNow['U'], dictNow['H'], dictNow['CZSL']))
    logger.info('Best GZSL|CZSL => S:{:.6f}; U:{:.6f}; H:{:.6f}; CZSL:{:.6f}'.format(
        best_record[0], best_record[1], best_record[2], best_record[3]))
    logger.info("-"*50)
    loss_record = 0.0
    netG.train()
    print("---------- One Epoch Ended ----------")
    if early_stop >= opt.early_stop:
        print(" ==> Early stop!")
        break


print(time.strftime('ending time:%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print('Dataset', opt.dataset)
print('the best GZSL seen accuracy is: ', best_record[0])
print('the best GZSL unseen accuracy is: ', best_record[1])
print('the best GZSL H is: ', best_record[2])
print('the best CZSL accuracy is: ', best_record[3])