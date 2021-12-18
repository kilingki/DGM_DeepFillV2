import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import glob
import pandas
import os
from os import listdir
import random
import gc
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class FaceDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=512):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.root = root
        trainlen = len(listdir(root))
        self.files = sorted(glob.glob("%s/*.png" % root))
        indexs = random.sample(range(0,trainlen), int(trainlen/100))
        self.files = [self.files[idx_f] for idx_f in indexs]

    def apply_random_mask(self, img):
        """Random mask image"""
        mask_num = np.random.randint(0, 299)
        mask_img = cv2.imread('dgm_masks/images_'+str(mask_num)+'.png')
        mask_img = torch.from_numpy(mask_img)
        masked_img = img.clone()
        mask_imgi = img.clone()
        for i in range(masked_img.shape[1]):
            for j in range(masked_img.shape[2]):
                if mask_img[i, j].all() == 1:
                    masked_img[:, i, j] = 1
                    mask_imgi[:, i, j] = 1
                else:
                    mask_imgi[:, i, j] = 0
        return masked_img, mask_imgi

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        masked_img, mask_img = self.apply_random_mask(img)
        return img, masked_img, mask_img

    def __len__(self):
        return len(self.files)

class Trainer(object):
    def __init__(self, epochs, batch_size, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self._build_model()
        self.binary_cross_entropy = torch.nn.BCELoss()
        self.p_loss = torch.nn.L1Loss() # Target
        transforms_ = [transforms.Resize((512, 512), Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataset = FaceDataset(root="dgm_dataset", transforms_=transforms_)
        self.root = dataset.root
        condition = str(self.epochs) + " epoch " + str(self.batch_size)+ " batch size " + str(int(self.learning_rate * 10000)) + " lr "
        
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.optimizer_G = torch.optim.Adam(self.gnet.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        print("Training " + condition + " ...")
    
    def _build_model(self):
        gnet = Generator()
        self.gnet = gnet.cuda()
        self.gnet.apply(weights_init_normal)
        self.gnet.train()

        print('Finish build model.')

    def train(self):
        date = '20211206'
        if os.path.isdir("w_path") == False: os.makedirs("w_path")
        if os.path.isdir("gen_imgs") == False: os.makedirs("gen_imgs")
        for epoch in tqdm.tqdm(range(self.epochs+1)):
            if (epoch == 3) or (epoch == 10):
                torch.save(self.gnet.state_dict(), "_".join(["w_path/model", str(epoch), ".pth"])) #Change this path

            if (epoch == 0) or (epoch == 2) or (epoch == 9):
                img_save_path = "gen_imgs/" + "_".join([str(epoch+1), str(self.batch_size), str(int(self.learning_rate * 10000))])
                if os.path.isdir(img_save_path) == False: os.makedirs(img_save_path)
      
            for batch_idx, (imgs, masked_imgs, mask_imgs) in enumerate(self.dataloader):
                Tensor = torch.cuda.FloatTensor
                imgs = Variable(imgs.type(Tensor))
                masked_imgs = Variable(masked_imgs.type(Tensor))
                mask_imgs = Variable(mask_imgs.type(Tensor))
                # Generator
                self.optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = self.gnet(masked_imgs)
                img_parts = imgs * mask_imgs
                gen_parts = gen_imgs * mask_imgs
                g_loss = self.p_loss(gen_parts, img_parts)
                
                # Save generated image
                if (epoch == 0) or (epoch == 2) or (epoch == 9):
                    batch_path = img_save_path + "/batch_"+str(batch_idx)
                    if os.path.isdir(batch_path) == False: os.makedirs(batch_path)
                    for idx1 in range(masked_imgs.size(0)):
                        gen_sample = torch.cat((masked_imgs[idx1, :, :, :].data, gen_parts[idx1, :, :, :].data, imgs[idx1, :, :, :].data), -2)
                        img_name = "/img" + str(idx1) + ".png"
                        save_image(gen_sample, batch_path + img_name, nrow=1, normalize=True)

                g_t_loss = g_loss
                g_t_loss.backward()
                self.optimizer_G.step()

            if (epoch % 10 == 9):
                print("[Epoch %d/%d] generator loss = %f & discrimminator loss = %f" % (epoch+1, self.epochs, round(float(g_t_loss), 6), round(float(d_t_loss), 6)))
            elif ((epoch == 2) or (epoch == 9)):
                print("[Epoch %d/%d] generator loss = %f" % (epoch+1, self.epochs, round(float(g_t_loss), 6)))

class Generator(nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()

        def downsampling(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsampling(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        self.model = nn.Sequential(
            *downsampling(channels, 512, normalize=False),  
            *downsampling(512, 512),
            *downsampling(512, 1024),
            *downsampling(1024, 2048),
            nn.Conv2d(2048, 8000, 1),
            *upsampling(8000, 2048),
            *upsampling(2048, 1024),
            *upsampling(1024, 512),
            *upsampling(512, 512),
            nn.Conv2d(512, channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

def main():

    batchSize = 4
    epochs = 10
    learningRate = 0.0002
    
    trainer = Trainer(epochs, batchSize, learningRate)
    trainer.train()


if __name__ == '__main__':
    main()