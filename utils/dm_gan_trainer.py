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
from utils.dmgan_utils import build_super_images, build_super_images2
from utils.dmgan_utils import weights_init, load_params, copy_G_params
from utils.model import G_DCGAN, G_NET
# from datasets import prepare_data
from utils.model import RNN_ENCODER, CNN_ENCODER

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

import logging

# ################# Text to image task############################ #
class condGANTrainer(object):
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

        #torch.cuda.set_device(cfg.GPU_ID)
        #cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE #
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
        logging.info('Load image encoder from:' + img_encoder_path)
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
        logging.info('Load text encoder from:' + cfg.TRAIN.NET_E)
        text_encoder.eval()

        # #######################generator and discriminators############## #
        netsD = []
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM ==1:
                from model import D_NET64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            # TODO: elif cfg.TREE.BRANCH_NUM > 3:
            netG = G_DCGAN()
            netsD = [D_NET(b_jcu=False)]
        else:
            from utils.model import D_NET64, D_NET128, D_NET256
            netG = G_NET()
            # netsD.append(D_NET128())
            # netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(D_NET256())
            # TODO: if cfg.TREE.BRANCH_NUM > 3:
        logging.info("Train with pretrained net G " + cfg.TRAIN.NET_G)
        print('number of trainable parameters =', count_parameters(netG))
        print('number of trainable parameters =', count_parameters(netsD[-1]))
        logging.info('number of trainable parameters =' + str(count_parameters(netG)))
        logging.info('number of trainable parameters =' + str(count_parameters(netsD[-1])))

        netG.apply(weights_init)
        # print(netG)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            # print(netsD[i])
        print('# of netsD', len(netsD))
        logging.info('# of netsD'+ str(len(netsD)))
        #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            logging.info('Load G from: ' + cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    logging.info('Load D from: '+ Dname)
                    state_dict = \
                        torch.load(Dname, map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)
        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.to(self.device) #cuda()
            image_encoder = image_encoder.to(self.device) #.cuda()
            netG.to(self.device) #.cuda()
            for i in range(len(netsD)):
                netsD[i].to(self.device) #.cuda()
        return [text_encoder, image_encoder, netG, netsD, epoch]

    def define_optimizers(self, netG, netsD, image_encoder):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(filter(lambda p: p.requires_grad, netsD[i].parameters()),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            # opt = hvd.DistributedOptimizer(
            #     opt,
            #     named_parameters=netsD[i].named_parameters())

            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))
        optimizer_image = optim.Adam(image_encoder.parameters(),
                                lr=cfg.TRAIN.ENCODER_LR,
                                betas=(0.5, 0.999))
        # optimizerG = hvd.DistributedOptimizer(
        #     optimizerG,
        #     named_parameters=netG.named_parameters())


        return optimizerG, optimizersD, optimizer_image

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.to(self.device) #.cuda()
            fake_labels = fake_labels.to(self.device) #.cuda()
            match_labels = match_labels.to(self.device) #.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, image_encoder, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(),
                '%s/netD%d.pth' % (self.model_dir, i))
        torch.save(image_encoder.state_dict(),
                   '%s/image_encoder_epoch_%d.pth' % (self.model_dir, epoch))
        print('Save G/Ds models.')
        logging.info('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, real_image, name='current'):
        import ipdb; ipdb.set_trace()
        # Save images
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask, cap_lens)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'% (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        # for i in range(len(netsD)):
        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)
        #print(real_image.type)

    def prepare_data(self, data):
        # imgs, captions, captions_lens, class_ids, keys = data
        imgs = data['img']
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
            captions = Variable(captions).to(self.device) #.cuda()
            sorted_cap_lens = Variable(sorted_cap_lens).to(self.device) #.cuda()
        else:
            captions = Variable(captions)
            sorted_cap_lens = Variable(sorted_cap_lens)

        return real_imgs, captions, sorted_cap_lens, class_ids, keys

    def train(self):
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD, optimizer_image = self.define_optimizers(netG, netsD, image_encoder)
        #multi-GPU
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
            noise, fixed_noise = noise.to(self.device), fixed_noise.to(self.device) #.cuda()
            
        writer = SummaryWriter(log_dir=os.path.join(cfg.LOG_DIR, datetime.now().strftime('%b%d_%H-%M-%S')+'_'+'ver_'+cfg.VERSION_NAME))
        # logdir = writer.file_writer.get_logdir()
        # savePath = join(logdir, savePath)
        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, captions, cap_lens, class_ids, keys = self.prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                netG = netG.to(self.device)
                fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask, cap_lens)

                #######################################################
                # (3) Update D network
                ######################################################
                # loss_dict = {'errsD':[], 'image_contrastive':[]}
                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD, log, log_dict = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],sent_emb, real_labels, fake_labels)
                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())
                    D_logs += log
                    # loss_dict['errsD'].append(errD)
                    # loss_dict['image_contrastive'].append(log_dict['image_contrastive'])

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                image_encoder.train()
                image_encoder.zero_grad()
                netG.zero_grad()
                errG_total, G_logs, G_log_dict = \
                    generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                   words_embs, sent_emb, match_labels, cap_lens, class_ids)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.item()
                if cfg.CONTRASTIVE.WORD_CONTRASTIVE:
                    image_contrastive_loss = log_dict['image_contrastive']
                    # c_loss_d = errD_total
                    # import ipdb; ipdb.set_trace()
                    errG_total += torch.tensor(image_contrastive_loss).to(self.device)
                    G_logs += 'image_c_loss: %.2f ' % image_contrastive_loss
                    # print("image_contrastive_loss : ", image_contrastive_loss, "errG_total : ", errG_total)
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                optimizer_image.step()
                image_encoder.eval()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)
                # loss_dict['errG'] = errG_total
                # self.save_img_results(netG, fixed_noise, sent_emb, words_embs, mask, image_encoder,
                #                       captions, cap_lens, epoch, imgs[-1], name='average')
                if gen_iterations % 100 == 0:
                    print('Epoch [{}/{}] Step [{}/{}]'.format(epoch, self.max_epoch, step,
                                                              self.num_batches) + ' ' + D_logs + ' ' + G_logs)
                    logging.info('Epoch [{}/{}] Step [{}/{}]'.format(epoch, self.max_epoch, step,
                                                              self.num_batches) + ' ' + D_logs + ' ' + G_logs)
                    for key in log_dict:
                        writer.add_scalar(key, log_dict[key], (epoch-1)*self.num_batches+gen_iterations)
                    for key in G_log_dict:
                        writer.add_scalar(key, G_log_dict[key], (epoch-1)*self.num_batches+gen_iterations)
                # save images
                if gen_iterations % 10000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    # self.save_img_results(netG, fixed_noise, sent_emb, words_embs, mask, image_encoder,
                    #                      captions, cap_lens, epoch, imgs[-1], name='average')
                    load_params(netG, backup_para)
                    #
                    # self.save_img_results(netG, fixed_noise, sent_emb,
                    #                       words_embs, mask, image_encoder,
                    #                       captions, cap_lens,
                    #                       epoch, name='current')
                # if gen_iterations % 1000 == 0:
                #    time.sleep(30)
                # if gen_iterations % 10000 == 0:
                #    time.sleep(160)
            end_t = time.time()

            print('''[%d/%d] Loss_D: %.2f Loss_G: %.2f Time: %.2fs''' % (
                epoch, self.max_epoch, errD_total.item(), errG_total.item(), end_t - start_t))
            logging.info('''[%d/%d] Loss_D: %.2f Loss_G: %.2f Time: %.2fs''' % (
                epoch, self.max_epoch, errD_total.item(), errG_total.item(), end_t - start_t))
            print(self.model_dir)
            print(cfg.TRAIN.NET_G)
            print('-' * 89)
            logging.info('-' * 89)
            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, image_encoder, epoch)
            writer.add_scalar('Total_errD', errD_total.item(), epoch)
            writer.add_scalar('Total_errG', errG_total.item(), epoch)

        self.save_model(netG, avg_param_G, netsD, image_encoder, self.max_epoch)

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
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

    def sampling(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.to(self.device) #.cuda()
            netG.eval()

            # load text encoder
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.to(self.device) #.cuda()
            text_encoder.eval()

            #load image encoder
            image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
            img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
            state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
            image_encoder.load_state_dict(state_dict)
            print('Load image encoder from:', img_encoder_path)
            image_encoder = image_encoder.to(self.device) #.cuda()
            image_encoder.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.to(self.device) #.cuda()

            model_dir = cfg.TRAIN.NET_G
            state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0
            R_count = 0
            R = np.zeros(30000)
            cont = True
            for ii in range(11):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                if (cont == False):
                    break
                for step, data in enumerate(self.data_loader, 0):
                    cnt += batch_size
                    if (cont == False):
                        break
                    if step % 100 == 0:
                       print('cnt: ', cnt)
                    # if step > 50:
                    #     break

                    imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask, cap_lens)
                    for j in range(batch_size):
                        s_tmp = '%s/single/%s' % (save_dir, keys[j])
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            #print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = -1
                        # for k in range(len(fake_imgs)):
                        im = fake_imgs[k][j].data.cpu().numpy()
                        # [-1, 1] --> [0, 255]
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = '%s_s%d_%d.png' % (s_tmp, k, ii)
                        im.save(fullpath)

                    _, cnn_code = image_encoder(fake_imgs[-1])

                    for i in range(batch_size):
                        mis_captions, mis_captions_len = self.dataset.get_mis_caption(class_ids[i])
                        hidden = text_encoder.init_hidden(99)
                        _, sent_emb_t = text_encoder(mis_captions, mis_captions_len, hidden)
                        rnn_code = torch.cat((sent_emb[i, :].unsqueeze(0), sent_emb_t), 0)
                        ### cnn_code = 1 * nef
                        ### rnn_code = 100 * nef
                        scores = torch.mm(cnn_code[i].unsqueeze(0), rnn_code.transpose(0, 1))  # 1* 100
                        cnn_code_norm = torch.norm(cnn_code[i].unsqueeze(0), 2, dim=1, keepdim=True)
                        rnn_code_norm = torch.norm(rnn_code, 2, dim=1, keepdim=True)
                        norm = torch.mm(cnn_code_norm, rnn_code_norm.transpose(0, 1))
                        scores0 = scores / norm.clamp(min=1e-8)
                        if torch.argmax(scores0) == 0:
                            R[R_count] = 1
                        R_count += 1

                    if R_count >= 30000:
                        sum = np.zeros(10)
                        np.random.shuffle(R)
                        for i in range(10):
                            sum[i] = np.average(R[i * 3000:(i + 1) * 3000 - 1])
                        R_mean = np.average(sum)
                        R_std = np.std(sum)
                        print("R mean:{:.4f} std:{:.4f}".format(R_mean, R_std))
                        cont = False





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
            text_encoder = text_encoder.to(self.device) #.cuda()
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
            netG.to(self.device) #.cuda()
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.to(self.device) #.cuda()
                cap_lens = cap_lens.to(self.device) #.cuda()
                for i in range(1):  # 16
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.to(self.device) #.cuda()
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
