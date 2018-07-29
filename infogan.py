import torch
import torchvision as tv
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os
import skimage.io as io

LR_D = 0.0002
LR_G = 0.001
BETA_1 = 0.5
BETA_2 = 0.999
NB_EPOCHS = 100
BATCH_SIZE = 32
DIR_DEST = os.path.join(".", "fake_images")




trainset = tv.datasets.MNIST(root='../../dataset/mnist', train=True,
                                        download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = tv.datasets.MNIST(root='../../dataset/mnist', train=False,
                                       download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)
class View(nn.Module):
    def __init__(self, dim):
        super(View, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.view((-1,) + self.dim)

class Encode(nn.Module):
    def __init__(self):
        super(Encode, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 1024, 7, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x):
        output = self.net(x)
        return output

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        output = self.net(x).view(-1, 1)
        return output


class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 12)
        )
    def forward(self, x):
        output = x.view(-1,1024)
        output = self.net(output)
        return output

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(74, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128*7*7),
            nn.BatchNorm1d(128*7*7),
            nn.ReLU(inplace=True),
            View((128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return output

class InfoGAN:
    def __init__(self):
        self.generator = G().cuda()
        self.encoder = Encode().cuda()
        self.discr = D().cuda()
        self.recog = Q().cuda()

    def generate_noise(self, batch_size, noise_var, dc_var, cc_var, dc_dim=10, cc_dim=2):
        idx = np.random.randint(10, size=batch_size)
        dc = np.zeros((batch_size, dc_dim))
        dc[range(batch_size), idx] = 1
        dc = torch.Tensor(dc)
        dc_var.data.copy_(dc)
        cc_var.data.uniform_(-1,1)
        noise_var.data.uniform_(-1,1)
        return idx

    def g(self, z, dc, cc):
        output = torch.cat((z, dc, cc), 1)
        output = self.generator(output)
        return output

    def q(self, x, dc_dim=10):
        output = self.encoder(x)
        output = self.recog(output)
        dc, cc = output[:,:dc_dim], output[:,dc_dim:]
        dc = torch.nn.Softmax()(dc)
        cc = torch.nn.Tanh()(cc)
        return dc, cc

    def d(self, x):
        output = self.encoder(x)
        output = self.discr(output)
        return output

    def d_q(self, x, dc_dim=10):
        output =  self.encoder(x)
        d = self.discr(output)
        output = self.recog(output)
        dc, cc = output[:,:dc_dim], output[:,dc_dim:]
        dc = torch.nn.Softmax()(dc)
        cc = torch.nn.Tanh()(cc)
        return d, dc, cc

    def train(self, loader, batch_size, nb_epochs=100):
        if os.path.exists(os.path.join(".", "fake_images")):
            pass
        else:
            os.mkdir(os.path.join(".", "fake_images"))
        real_x = torch.FloatTensor(batch_size, 1, 28, 28).cuda()
        dc = torch.FloatTensor(batch_size, 10).cuda()
        cc = torch.FloatTensor(batch_size, 2).cuda()
        label = torch.FloatTensor(batch_size).cuda()
        noise = torch.FloatTensor(batch_size, 62).cuda()

        real_x = Variable(real_x)
        label = Variable(label, requires_grad=False)
        dc = Variable(dc)
        cc = Variable(cc)
        noise = Variable(noise)

        criterionD = nn.BCELoss().cuda()
        criterionQ_dis = nn.CrossEntropyLoss().cuda()
        criterionQ_con = nn.MSELoss().cuda()

        g_optim = optim.Adam(
            [{'params':self.generator.parameters()},{'params':self.recog.parameters()}],
            lr=LR_G,
            betas=(BETA_1, BETA_2)
        )
        d_optim = optim.Adam(
            [{'params':self.encoder.parameters()},{'params':self.discr.parameters()}],
            lr=LR_G,
            betas=(BETA_1, BETA_2)
        )

        for epoch in range(NB_EPOCHS):
            for batch_index, batch_data in enumerate(loader, 0):
                d_optim.zero_grad()

                # for real data
                X_real, _ = batch_data
                real_x.data.copy_(X_real)
                label.data.fill_(1)
                d_real = self.d(real_x)
                d_real = d_real.squeeze()
                d_loss_real = criterionD(d_real,label)
                d_loss_real.backward()

                # for fake data
                idx = self.generate_noise(batch_size, noise, dc,cc)
                idx = torch.Tensor(idx)
                X_fake = self.g(noise, dc, cc)
                d_fake = self.d(X_fake)
                d_fake = d_fake.squeeze()
                label.data.fill_(0)
                d_loss_fake = criterionD(d_fake, label)
                d_loss_fake.backward()

                d_loss = d_loss_real + d_loss_fake
                d_optim.step()

                g_optim.zero_grad()
                idx = self.generate_noise(batch_size, noise, dc,cc)
                idx = torch.Tensor(idx)
                X_fake = self.g(noise, dc, cc)
                d_fake, dc_fake, cc_fake = self.d_q(X_fake)
                d_fake = d_fake.squeeze()
                label.data.fill_(1)
                g_loss_fake = criterionD(d_fake, label)
                target = Variable(idx.long().cuda())
                dc_loss = criterionQ_dis(dc_fake, target)
                cc_loss = criterionQ_con(cc_fake, cc)

                g_loss = g_loss_fake + dc_loss + cc_loss
                g_loss.backward()
                g_optim.step()

            print("Epoch %d"%(epoch), "d_loss %.4f"%(d_loss.data.cpu().numpy()), "g_loss %.4f"%(g_loss.data.cpu().numpy()))
            # save images
            for i in range(batch_size):
                curr_image = 255 * X_fake[i,0,:,:].cpu().data.numpy()
                curr_image = curr_image.astype(np.int32)
                curr_label = dc_fake[i].cpu().data.numpy().argmax()
                fname = "ep%d-dc%d-idx%d.png"%(epoch, curr_label, i)
                io.imsave(os.path.join(DIR_DEST, fname), curr_image)

igan = InfoGAN()
igan.train(trainloader,BATCH_SIZE)
