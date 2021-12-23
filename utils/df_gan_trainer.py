from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image

from miscc.config import cfg
from utils.dmgan_utils import mkdir_p
# from utils.dmgan_utils import build_super_images, build_super_images2
from utils.dmgan_utils import weights_init, load_params, copy_G_params
# from utils.model import G_DCGAN, G_NET
from utils.DAMSM import RNN_ENCODER
# from datasets import prepare_data
from utils.model import CNN_ENCODER #RNN_ENCODER,
from utils.dfgan_model import NetG, NetD

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss
import os
import time
import numpy as np
import sys
import tensorboard
from tensorboardX import SummaryWriter
from datetime import datetime
import horovod.torch as hvd
import sys
import random
import pprint
import dateutil.tz
import torchvision.transforms as transforms
import torchvision.utils as vutils

import logging


# ################# Text to image task############################ #
class DFGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword, dataset):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logging.basicConfig(level=logging.DEBUG, filename=output_dir+"/logfile.log", filemode="a+",
                            format="%(asctime)-15s %(levelname)-8s %(message)s")
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        # torch.cuda.set_device(cfg.GPU_ID)
        # cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE  #
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.dataset = dataset
        self.num_batches = len(self.data_loader)
        # self.NET_E = cfg.TRAIN.RNN_ENCODER
        # self.NET_G =
        # self.
        self.device = 'cuda:' + str(0) if torch.cuda.device_count() > 0 else 'cpu'

    def build_models(self):
        def count_parameters(model):
            total_param = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    num_param = np.prod(param.size())
                    if param.dim() > 1:
                        print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
                    else:
                        print(name, ':', num_param)
                    total_param += num_param
            return total_param

        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        logging.info('Load image encoder from:'+img_encoder_path)
        image_encoder.eval()

        text_encoder = \
            RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        logging.info('Load text encoder from:'+ cfg.TRAIN.NET_E)
        text_encoder.eval()
        import ipdb; ipdb.set_trace()

        # #######################generator and discriminators############## #
        netG = NetG(cfg.TRAIN.NF, cfg.GAN.Z_DIM).to(self.device)
        netD = NetD(cfg.TRAIN.NF).to(self.device)

        print('number of trainable parameters =', count_parameters(netG))
        print('number of trainable parameters =', count_parameters(netD))
        logging.info('number of trainable parameters ='+ str(count_parameters(netG)))
        logging.info('number of trainable parameters =' + str(count_parameters(netD)))

        netG.apply(weights_init)
        netD.apply(weights_init)
        #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            # copy_state_dict = {}
            # for key in netG.state_dict():
            #     if key in state_dict:
            #
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            logging.info('Load G from: '+ cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                s_tmp = Gname[:Gname.rfind('/')]
                Dname = Gname.replace("netG", "netD") #("netG_epoch_", "netD_epoch")
                state_dict = torch.load(Dname, map_location=lambda storage, loc: storage)
                netD.load_state_dict(state_dict)
        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.to(self.device)  # cuda()
            image_encoder = image_encoder.to(self.device)  # .cuda()
            netG.to(self.device)  # .cuda()
            netD.to(self.device)

        return [text_encoder, image_encoder, netG, netD, epoch]

    def define_optimizers(self, netG, netD):
        optimizerD = optim.Adam(netD.parameters(),
                                lr=cfg.TRAIN.DISCRIMINATOR_LR,
                                betas=(0.0, 0.9))
        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.0, 0.9))
        # optimizerG = hvd.DistributedOptimizer(
        #     optimizerG,
        #     named_parameters=netG.named_parameters())

        return optimizerG, optimizerD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.to(self.device)  # .cuda()
            fake_labels = fake_labels.to(self.device)  # .cuda()
            match_labels = match_labels.to(self.device)  # .cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, netD, epoch):
        # backup_para = copy_G_params(netG)
        # load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
                   '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        # load_params(netG, backup_para)
        torch.save(netD.state_dict(), '%s/netD_epoch%d.pth' % (self.model_dir, epoch))
        print('Save G/Ds models.')
        logging.info('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    # def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
    #                      image_encoder, captions, cap_lens,
    #                      gen_iterations, real_image, name='current'):
    #     # Save images
    #     fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask, cap_lens)
    #     for i in range(len(attention_maps)):
    #         if len(fake_imgs) > 1:
    #             img = fake_imgs[i + 1].detach().cpu()
    #             lr_img = fake_imgs[i].detach().cpu()
    #         else:
    #             img = fake_imgs[0].detach().cpu()
    #             lr_img = None
    #         attn_maps = attention_maps[i]
    #         att_sze = attn_maps.size(2)
    #         img_set, _ = \
    #             build_super_images(img, captions, self.ixtoword,
    #                                attn_maps, att_sze, lr_imgs=lr_img)
    #         if img_set is not None:
    #             im = Image.fromarray(img_set)
    #             fullpath = '%s/G_%s_%d_%d.png' % (self.image_dir, name, gen_iterations, i)
    #             im.save(fullpath)
    #
    #     # for i in range(len(netsD)):
    #     i = -1
    #     img = fake_imgs[i].detach()
    #     region_features, _ = image_encoder(img)
    #     att_sze = region_features.size(2)
    #     _, _, att_maps = words_loss(region_features.detach(),
    #                                 words_embs.detach(),
    #                                 None, cap_lens,
    #                                 None, self.batch_size)
    #     img_set, _ = \
    #         build_super_images(fake_imgs[i].detach().cpu(),
    #                            captions, self.ixtoword, att_maps, att_sze)
    #     if img_set is not None:
    #         im = Image.fromarray(img_set)
    #         fullpath = '%s/D_%s_%d.png' \
    #                    % (self.image_dir, name, gen_iterations)
    #         im.save(fullpath)
    #     # print(real_image.type)

    def prepare_data(self, data):
        # imgs, captions, captions_lens, class_ids, keys = data
        if 'img' in data:
            imgs = data['img']
        elif 'gen_img' in data:
            imgs = data['gen_img']
        else:
            imgs = []
        captions = data['caps']
        captions_lens = data['cap_len']
        class_ids = data['cls_id']
        keys = data['key']
        sentence_index = data['sent_ix']

        # sort data by the length in a decreasing order
        sorted_cap_lens, sorted_cap_indices = \
            torch.sort(captions_lens, 0, True)

        real_imgs = []
        for i in range(len(imgs)):
            imgs[i] = imgs[i][sorted_cap_indices]
            if cfg.CUDA:
                real_imgs.append(Variable(imgs[i]).to(self.device))
            else:
                real_imgs.append(Variable(imgs[i]))

        captions = captions[sorted_cap_indices].squeeze()
        class_ids = class_ids[sorted_cap_indices].numpy()
        # sent_indices = sent_indices[sorted_cap_indices]
        keys = [keys[i] for i in sorted_cap_indices.numpy()]
        # print('keys', type(keys), keys[-1])  # list
        if cfg.CUDA:
            captions = Variable(captions).to(self.device)  # .cuda()
            sorted_cap_lens = Variable(sorted_cap_lens).to(self.device)  # .cuda()
        else:
            captions = Variable(captions)
            sorted_cap_lens = Variable(sorted_cap_lens)

        return real_imgs, captions, sorted_cap_lens, class_ids, keys, sentence_index

    def train(self):
        text_encoder, image_encoder, netG, netD, start_epoch = self.build_models()
        # avg_param_G = copy_G_params(netG)
        optimizerG, optimizerD = self.define_optimizers(netG, netD)
        # multi-GPU
        # hvd.broadcast_parameters(
        #     netG.state_dict(),
        #     root_rank=0)
        # for i in len(netsD):
        #     hvd.broadcast_parameters(
        #         netsD[i].state_dict(),
        #         root_rank=0)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.to(self.device), fixed_noise.to(self.device)  # .cuda()

        writer = SummaryWriter(log_dir=os.path.join(cfg.LOG_DIR, datetime.now().strftime(
            '%b%d_%H-%M-%S') + '_' + 'ver_' + cfg.VERSION_NAME))
        # logdir = writer.file_writer.get_logdir()
        # savePath = join(logdir, savePath)
        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            for step, data in enumerate(self.data_loader, 0):
                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                # data = data_iter.next()
                imgs, captions, cap_lens, class_ids, keys, sent_idx = self.prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                # import ipdb; ipdb.set_trace()
                imgs = imgs[0]
                real_features = netD(imgs)
                output = netD.COND_DNET(real_features, sent_emb)
                errD_real = torch.nn.ReLU()(1.0-output).mean()

                output = netD.COND_DNET(real_features[:(batch_size - 1)], sent_emb[1:batch_size])
                errD_mismatch = torch.nn.ReLU()(1.0 + output).mean()

                # synthesize fake images
                noise = torch.randn(batch_size, 100)
                noise = noise.to(self.device)
                fake = netG(noise, sent_emb)

                # G does not need update with D
                fake_features = netD(fake.detach())

                errD_fake = netD.COND_DNET(fake_features, sent_emb)
                errD_fake = torch.nn.ReLU()(1.0 + errD_fake).mean()

                errD = errD_real + (errD_fake + errD_mismatch) / 2.0

                step += 1
                gen_iterations += 1
                optimizerD.zero_grad()
                optimizerG.zero_grad()
                errD.backward()
                optimizerD.step()

                # MA-GP
                interpolated = (imgs.data).requires_grad_()
                sent_inter = (sent_emb.data).requires_grad_()
                features = netD(interpolated)
                out = netD.COND_DNET(features, sent_inter)
                grads = torch.autograd.grad(outputs=out,
                                            inputs=(interpolated, sent_inter),
                                            grad_outputs=torch.ones(out.size()).cuda(),
                                            retain_graph=True,
                                            create_graph=True,
                                            only_inputs=True)
                grad0 = grads[0].view(grads[0].size(0), -1)
                grad1 = grads[1].view(grads[1].size(0), -1)
                grad = torch.cat((grad0, grad1), dim=1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm) ** 6)
                d_loss = 2.0 * d_loss_gp
                optimizerD.zero_grad()
                optimizerG.zero_grad()
                d_loss.backward()
                optimizerD.step()

                # update G
                features = netD(fake)
                output = netD.COND_DNET(features, sent_emb)
                errG = - output.mean()
                optimizerG.zero_grad()
                optimizerD.zero_grad()
                errG.backward()
                optimizerG.step()



                result_dict = {'errD':errD, 'errG':errG}
                # D_logs = ''
                # G_logs = ''
                #
                if gen_iterations % 100 == 0:
                    print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
                          % (epoch, cfg.TRAIN.MAX_EPOCH, step, self.num_batches, errD.item(), errG.item()))
                    logging.info('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
                                 % (epoch, cfg.TRAIN.MAX_EPOCH, step, self.num_batches, errD.item(), errG.item()))
                #     print('Epoch [{}/{}] Step [{}/{}]'.format(epoch, self.max_epoch, step,
                #                                               self.num_batches) + ' ' + D_logs + ' ' + G_logs)
                #     for key in log_dict:
                #         writer.add_scalar(key, log_dict[key], (epoch - 1) * self.num_batches + gen_iterations)
                #     for key in G_log_dict:
                #         writer.add_scalar(key, G_log_dict[key], (epoch - 1) * self.num_batches + gen_iterations)
                    for key in result_dict:
                        writer.add_scalar(key, result_dict[key], (epoch - 1) * self.num_batches + gen_iterations)
                # save images

            end_t = time.time()

            print('''[%d/%d] Loss_D: %.2f Loss_G: %.2f Time: %.2fs''' % (
                epoch, self.max_epoch, errD.item(), errG.item(), end_t - start_t))
            logging.info('''[%d/%d] Loss_D: %.2f Loss_G: %.2f Time: %.2fs''' % (
                epoch, self.max_epoch, errD.item(), errG.item(), end_t - start_t))
            print('-' * 89)
            logging.info('-' * 89)
            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, netD, epoch)
            writer.add_scalar('Total_errD', errD.item(), epoch)
            writer.add_scalar('Total_errG', errG.item(), epoch)

            vutils.save_image(fake.data,
                              '%s/fake_samples_epoch_%03d.png' % (self.image_dir, epoch),
                              normalize=True)

            # if epoch % 10 == 0:
            #     torch.save(netG.state_dict(), 'models/%s/netG_%03d.pth' % (cfg.CONFIG_NAME, epoch))
            #     torch.save(netD.state_dict(), 'models/%s/netD_%03d.pth' % (cfg.CONFIG_NAME, epoch))

        self.save_model(netG, netD, self.max_epoch)

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' % \
                    (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling(self, text_encoder, netG, dataloader, device):
        import ipdb; ipdb.set_tra
        model_dir = cfg.TRAIN.NET_G
        split_dir = 'valid'
        # Build and load the generator
        netG.load_state_dict(torch.load('models/%s/netG.pth' % (cfg.CONFIG_NAME)))
        netG.eval()

        batch_size = cfg.TRAIN.BATCH_SIZE
        s_tmp = model_dir
        save_dir = '%s/%s' % (s_tmp, split_dir)
        mkdir_p(save_dir)
        cnt = 0
        for i in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
            for step, data in enumerate(dataloader, 0):
                imags, captions, cap_lens, class_ids, keys, sent_idx = prepare_data(data)
                cnt += batch_size
                if step % 100 == 0:
                    print('step: ', step)
                # if step > 50:
                #     break
                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                #######################################################
                # (2) Generate fake images
                ######################################################
                with torch.no_grad():
                    noise = torch.randn(batch_size, 100)
                    noise = noise.to(device)
                    fake_imgs = netG(noise, sent_emb)
                for j in range(batch_size):
                    s_tmp = '%s/single/%s' % (save_dir, keys[j])
                    folder = s_tmp[:s_tmp.rfind('/')]
                    if not os.path.isdir(folder):
                        print('Make a new folder: ', folder)
                        mkdir_p(folder)
                    im = fake_imgs[j].data.cpu().numpy()
                    # [-1, 1] --> [0, 255]
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    fullpath = '%s_%3d.png' % (s_tmp, i)
                    im.save(fullpath)

    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.to(self.device)  # .cuda()
            text_encoder.eval()

            # the path to save generated images
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.to(self.device)  # .cuda()
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.to(self.device)  # .cuda()
                cap_lens = cap_lens.to(self.device)  # .cuda()
                for i in range(1):  # 16
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.to(self.device)  # .cuda()
                    #######################################################
                    # (1) Extract text embeddings
                    ######################################################
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = (captions == 0)
                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask, cap_lens)
                    # G attention
                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            # print('im', im.shape)
                            im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)
