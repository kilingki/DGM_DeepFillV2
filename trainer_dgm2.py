import os
import time
import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from collections import OrderedDict

import network_dgm
import train_dataset_dgm
import utils_dgm

def WGAN_trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    #save_folder = opt.save_path
    save_folder = './pretrained_model'
    sample_folder = opt.sample_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # Build networks
    generator = utils_dgm.create_generator(opt)
    patch_discriminator = utils_dgm.create_patch_discriminator(opt)
    maskaware_discriminator = utils_dgm.create_maskaware_discriminator(opt)
    perceptualnet = utils_dgm.create_perceptualnet()

    # Loss functions
    L1Loss = nn.L1Loss()
    MSELoss = nn.MSELoss()
    BCELoss = nn.BCELoss()
    
    # Optimizers
    #optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_pd = torch.optim.Adam(patch_discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_md = torch.optim.Adam(maskaware_discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(lr_in, optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = lr_in * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the two-stage generator model
    def save_model_generator(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_name)
                print('The trained G model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_name)
                print('The trained G model is successfully saved at epoch %d' % (epoch))
                
    # Save the PATCH dicriminator model
    def save_model_patch_discriminator(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'deepfillv2_WGAN_pD_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_name)
                print('The trained pD model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_name)
                print('The trained pD model is successfully saved at epoch %d' % (epoch))
    
    # Save the MASK-AWARE dicriminator model
    def save_model_maskaware_discriminator(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'deepfillv2_WGAN_mD_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_name)
                print('The trained mD model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_name)
                print('The trained mD model is successfully saved at epoch %d' % (epoch))
    
    # load the model
    def load_model(net, epoch, opt, type='G'):
        """Save the model at "checkpoint_interval" and its multiple"""
        if type == 'G':
            model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        else:
            model_name = 'deepfillv2_WGAN_D_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        
        pretrained_dict = torch.load(model_name)
        
        '''new_state_dict = OrderedDict()
        for n, v in pretrained_dict.items():
            #name = n.replace("module.","") # .module이 중간에 포함된 형태라면 (".module","")로 치환
            name = 'module.' + n
            new_state_dict[name] = v
        
        net.load_state_dict(new_state_dict)'''
        net.load_state_dict(pretrained_dict)
        
    load_model(generator, opt.epochs, opt)
    
    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        patch_discriminator = nn.DataParallel(patch_discriminator)
        maskaware_discriminator = nn.DataParallel(maskaware_discriminator)
        perceptualnet = nn.DataParallel(perceptualnet)
        
        generator = generator.cuda()
        patch_discriminator = patch_discriminator.cuda()
        maskaware_discriminator = maskaware_discriminator.cuda()
        perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        patch_discriminator = patch_discriminator.cuda()
        maskaware_discriminator = maskaware_discriminator.cuda()
        perceptualnet = perceptualnet.cuda()
    
    generator.eval()
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = train_dataset_dgm.InpaintDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True, drop_last=True)
    
    # ----------------------------------------
    #            Training
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()
    
    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # Training loop
    for epoch in range(opt.resume_epoch, opt.epochs):
        s_t = time.time()
        for batch_idx, (img, height, width) in enumerate(dataloader):

            img = img.cuda()
            # set the same free form masks for each batch
            mask = torch.empty(img.shape[0], 1, img.shape[2], img.shape[3]).cuda()
            for i in range(opt.batch_size):
                mask[i] = torch.from_numpy(train_dataset_dgm.InpaintDataset.random_ff_mask(
                                                shape=(height[0], width[0])).astype(np.float32)).cuda()
            
            # LSGAN vectors
            valid = Tensor(np.ones((img.shape[0], 1, height[0]//32, width[0]//32)))
            fake = Tensor(np.zeros((img.shape[0], 1, height[0]//32, width[0]//32)))
            zero = Tensor(np.zeros((img.shape[0], 1, height[0]//32, width[0]//32)))

            
            ###########################################################################
            ### Train Patch Discriminator #############################################
            ###########################################################################
            optimizer_pd.zero_grad()

            # Generator output
            with torch.no_grad():
                _, second_out = generator(img, mask) # coarse(1st) / refine(2nd)
            #second_out = second_out.detach()

            # forward propagation
            #first_out_wholeimg = img * (1 - mask) + first_out * mask        # in range [0, 1]
            second_out_wholeimg = img * (1 - mask) + second_out * mask      # in range [0, 1]

            # Fake samples
            fake_scalar = patch_discriminator(second_out_wholeimg.detach(), mask)
            # True samples
            true_scalar = patch_discriminator(img, mask)
            
            # Loss and optimize
            loss_fake = -torch.mean(torch.min(zero, -valid-fake_scalar))
            loss_true = -torch.mean(torch.min(zero, -valid+true_scalar))
            
            # Overall Loss and optimize
            loss_pD = 0.5 * (loss_fake + loss_true)
            loss_pD.backward()
            optimizer_pd.step()
            
            ###########################################################################
            ### Train Mask-aware Discriminator ########################################
            ###########################################################################
            optimizer_md.zero_grad()
            
            # Fake samples
            fake_scalar = maskaware_discriminator(second_out_wholeimg.detach())
            # True samples
            true_scalar = maskaware_discriminator(img)
            
            # Loss and optimize
            loss_fake = BCELoss(fake_scalar, mask)
            loss_true = BCELoss(true_scalar, Tensor(np.zeros(fake_scalar.shape)).detach())
            
            # Overall Loss and optimize
            loss_mD = 0.5 * (loss_fake + loss_true)
            loss_mD.backward()
            optimizer_md.step()
            
            '''#######################################################################
            ### Train Generator ###################################################
            #######################################################################
            optimizer_g.zero_grad()

            # L1 Loss
            first_L1Loss = (first_out - img).abs().mean()
            second_L1Loss = (second_out - img).abs().mean()
            
            # GAN Loss (Patch)
            fake_scalar1 = patch_discriminator(second_out_wholeimg, mask)
            GAN_Loss_Patch = -torch.mean(fake_scalar1)
            
            # GAN Loss (Mask-aware)
            fake_scalar2 = maskaware_discriminator(second_out_wholeimg)
            GAN_Loss_Mask = BCELoss(fake_scalar2, mask)
            
            # Get the deep semantic feature maps, and compute Perceptual Loss
            img_featuremaps = perceptualnet(img)    # feature maps
            second_out_featuremaps = perceptualnet(second_out)
            second_PerceptualLoss = L1Loss(second_out_featuremaps, img_featuremaps)

            # Compute losses
            loss = opt.lambda_l1 * first_L1Loss + opt.lambda_l1 * second_L1Loss \
                    + opt.lambda_perceptual * second_PerceptualLoss \
                    #+ opt.lambda_gan2 * GAN_Loss_Mask
                    #+ opt.lambda_gan1 * GAN_Loss_Patch
                    
            loss.backward()
            optimizer_g.step()'''

            # Print log
            if batch_idx % 100 == 0:
                #print("\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f]" %
                #    ((epoch + 1), opt.epochs, batch_idx, len(dataloader), first_L1Loss.item(), second_L1Loss.item()))
                #print("\r[pD Loss: %.5f] [mD Loss: %.5f] [G Loss_P: %.5f] [G Loss_M: %.5f] [P Loss: %.5f]" %
                #    (loss_pD.item(), loss_mD.item(), GAN_Loss_Patch.item(), GAN_Loss_Mask.item(), second_PerceptualLoss.item()))
                
                print("\r[Epoch %d/%d] [Batch %d/%d] [pD Loss: %.5f] [mD Loss: %.5f]" %
                    ((epoch + 1), opt.epochs, batch_idx, len(dataloader), loss_pD.item(), loss_mD.item()))
                
                print('100 batches take', time.time()-s_t)
                s_t = time.time()
                
            masked_img = img * (1 - mask) + mask
            mask = torch.cat((mask, mask, mask), 1)
            if (batch_idx + 1) % 40 == 0:
                img_list = [img, mask, masked_img, second_out]
                name_list = ['gt', 'mask', 'masked_img', 'second_out']
                utils_dgm.save_sample_png(sample_folder = sample_folder, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

        # Learning rate decrease
        #adjust_learning_rate(opt.lr_g, optimizer_g, (epoch + 1), opt)
        adjust_learning_rate(opt.lr_d, optimizer_pd, (epoch + 1), opt)
        adjust_learning_rate(opt.lr_d, optimizer_md, (epoch + 1), opt)
        
        # Save the model
        #save_model_generator(generator, (epoch + 1), opt)
        save_model_patch_discriminator(patch_discriminator, (epoch + 1), opt)
        save_model_maskaware_discriminator(maskaware_discriminator, (epoch + 1), opt)
        
        ### Sample data every epoch
        #if (epoch + 1) % 1 == 0:
        #    img_list = [img, mask, masked_img, first_out, second_out]
        #    name_list = ['gt', 'mask', 'masked_img', 'first_out', 'second_out']
        #    utils_dgm.save_sample_png(sample_folder = sample_folder, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
