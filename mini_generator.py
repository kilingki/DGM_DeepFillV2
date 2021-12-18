import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import glob
import os
import random
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class HumanDataset(Dataset):
    def __init__(self, root, transforms_):
        self.transform = transforms.Compose(transforms_)
        self.root = root
        trainlen = len(listdir(root))
        self.files = sorted(glob.glob("%s/*.png" % root))
        indexs = random.sample(range(0,trainlen), int(trainlen/100))
        self.files = [self.files[idx_f] for idx_f in indexs]

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        mask_num = np.random.randint(0, 299)
        mask_img = cv2.imread('dgm_masks/images_'+str(mask_num)+'.png') #Change this path
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

    def __len__(self):
        return len(self.files)

class MiniGenerator(nn.Module):
    def __init__(self, channels=3):
        super(MiniGenerator, self).__init__()

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

class MiniModel(object):
    def __init__(self, epochs = 10, batch_size = 4, lr = 0.002):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        gnet = MiniGenerator()
        self.gnet = gnet.cuda()
        # train
        if os.path.exist('mini_model.pth'):
            self.gnet.load_state_dict(torch.load('mini_model.pth'))
        else:
            self._build_model()
            transforms_ = [transforms.Resize((512, 512), Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            dataset = HumanDataset(root="dgm_dataset", transforms_=transforms_)
            self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self.optimizer_G = torch.optim.Adam(self.gnet.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
            self.train()
        # test
        self._test_build_model()
    
    def _build_model(self):
        self.p_loss = torch.nn.L1Loss() # Target
        self.gnet.apply(weights_init_normal)
        self.gnet.train()

    def train(self):
        for epoch in tqdm.tqdm(range(self.epochs+1)):
            for batch_idx, (imgs, masked_imgs, mask_imgs) in enumerate(self.dataloader):
                Tensor = torch.cuda.FloatTensor
                imgs = Variable(imgs.type(Tensor))
                masked_imgs = Variable(masked_imgs.type(Tensor))
                mask_imgs = Variable(mask_imgs.type(Tensor))
                # Generator
                self.optimizer_G.zero_grad()
                gen_imgs = self.gnet(masked_imgs)
                img_parts = imgs * mask_imgs
                gen_parts = gen_imgs * mask_imgs
                g_loss = self.p_loss(gen_parts, img_parts)
                g_loss.backward()
                self.optimizer_G.step()
        torch.save(self.gnet.state_dict(), 'mini_model.pth') #Change this path
    
    def _test_build_model(self):
        self.gnet.eval()

    def test(self, img):
        #Tensor = torch.cuda.FloatTensor
        #img_target = Variable(img.type(Tensor))

        gen_mask = self.gnet(img)
        return gen_mask
