import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
from miscc.config import cfg
from PIL import Image

import numpy as np
import os
import time

#########################################
# STUDENT IMPLEMENTATION
from utils.dm_gan_trainer import condGANTrainer
from utils.model import G_DCGAN, G_NET
from utils.dmgan_utils import build_super_images, build_super_images2
from utils.df_gan_trainer import DFGANTrainer
from utils.dfgan_model import NetG, NetD
#########################################
import logging
#################################################
# DO NOT CHANGE 
from utils.model import RNN_ENCODER, CNN_ENCODER, GENERATOR, DISCRIMINATOR
#################################################
device = 'cuda:' + str(0) if torch.cuda.device_count() > 0 else 'cpu'
def weights_init(m):
    imgs = data['img']
    captions = data['caps']
    captions_lens = data['cap_len']
    class_ids = data['cls_id']
    keys = data['key']
    sentence_idx = data['sent_ix']
    # wrong_caps = data['wrong_caps']
    # wrong_caps_len = data['wrong_cap_len']
    # wrong_cls_id = data['wrong_cls_id']

    # sort data by the length in a decreasing order
    # the reason of sorting data can be found in https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html
    sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)
    real_imgs = []
    original_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).to(device))  # F.interpolate(imgs[i],scale_factor=2)).cuda())
            original_imgs.append(Variable(imgs[i]).to(device))
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    keys = [keys[i] for i in sorted_cap_indices.numpy()]

    if cfg.CUDA:
        captions = Variable(captions).to(device)
        sorted_cap_lens = Variable(sorted_cap_lens).to(device)
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)
    #################################################
    # TODO
    # this part can be different, depending on which algorithm is used
    #################################################

    
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
                
class trainer(object):
    def __init__(self, output_dir, train_dataset, train_dataloader, test_dataset, test_dataloader, train_ixtoword, test_ixtoword):
        self.output_dir = output_dir
        self.image_dir = os.path.join(self.output_dir, 'Image')
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        
        self.batch_size = cfg.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        
        self.n_words = train_dataset.n_words # size of the dictionary
        self.train_ixtoword = train_ixtoword
        self.test_ixtoword = test_ixtoword
        self.train_here = False
        if cfg.GAN.TYPE=='DM_GAN':
            self.gan_trainer = condGANTrainer(output_dir, train_dataloader,self.n_words, self.train_ixtoword, train_dataset)
            self.gan_test = condGANTrainer(output_dir, test_dataloader, self.n_words, self.test_ixtoword, test_dataset)
        elif cfg.GAN.TYPE == 'DF_GAN':
            self.gan_trainer = DFGANTrainer(output_dir, train_dataloader, self.n_words, self.train_ixtoword,
                                              train_dataset)
            self.gan_test = DFGANTrainer(output_dir, test_dataloader, self.n_words, self.test_ixtoword, test_dataset)
        elif cfg.GAN.TYPE == 'MY_GAN':
            self.train_here = True

        self.device = 'cuda:' + str(0) if torch.cuda.device_count() > 0 else 'cpu'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logging.basicConfig(level=logging.DEBUG, filename=output_dir + "/logfile.log", filemode="a+",
                            format="%(asctime)-15s %(levelname)-8s %(message)s")
    
    def prepare_data(self, data):
        """
        Prepares data given by dataloader
        e.g., x = Variable(x).cuda()
        """
        if 'img' in data:
            imgs = data['img']
        elif 'gen_img' in data:
            imgs = data['gen_img']
        else:
            imgs = None
        captions = data['caps']
        captions_lens = data['cap_len']
        class_ids = data['cls_id']
        keys = data['key']
        sentence_idx = data['sent_ix']
        
        # sort data by the length in a decreasing order
        # the reason of sorting data can be found in https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html 
        sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)
        
        #################################################
        # TODO
        # this part can be different, depending on which algorithm is used
        # imgs = imgs[sorted_cap_indices]
        # if cfg.CUDA:
        #     imgs = Variable(imgs).cuda()
        #################################################
        if imgs is not None:
            real_imgs = []
            for i in range(len(imgs)):
                imgs[i] = imgs[i][sorted_cap_indices]
                if cfg.CUDA:
                    real_imgs.append(Variable(imgs[i]).to(self.device))
                else:
                    real_imgs.append(Variable(imgs[i]))
        else:
            real_imgs = []

        captions = captions[sorted_cap_indices].squeeze()
        class_ids = class_ids[sorted_cap_indices].numpy()
        keys = [keys[i] for i in sorted_cap_indices.numpy()]

        if cfg.CUDA:
            captions = Variable(captions).to(self.device)
            sorted_cap_lens = Variable(sorted_cap_lens).to(self.device)
        else:
            captions = Variable(captions)
            sorted_cap_lens = Variable(sorted_cap_lens)

        return [real_imgs, captions, sorted_cap_lens, class_ids, keys, sentence_idx]
    
    def train(self):
        """
        e.g., for epoch in range(cfg.TRAIN.MAX_EPOCH):
                  for step, data in enumerate(self.train_dataloader, 0):
                      x = self.prepare_data()
                      .....
        """
        #################################################
        # TODO: Implement text to image synthesis
        if self.train_here:
            self.build_models()
            optimizerG, optimizerD = self.get_optimizers()
            start_time = time.time()
            for epoch in range(self.start_epoch + 1, cfg.TRAIN.MAX_EPOCH + 1):
                for step, data in enumerate(self.train_dataloader):
                    imgs, captions, cap_lens, class_ids, keys, sent_idx = self.prepare_data(data)
                    imgs = imgs[0]
                    # Embeddings
                    hidden = self.text_encoder.init_hidden(self.batch_size)
                    words_emb, sent_emb = self.text_encoder(captions, cap_lens, hidden)
                    words_emb, sent_emb = words_emb.detach(), sent_emb.detach()
                    # Real images
                    real_features = self.netD(imgs)
                    output = self.netD.cond_net(real_features, sent_emb)
                    errD_real = torch.nn.ReLU()(1.0 - output).mean()
                    output = self.netD.cond_net(real_features[:(self.batch_size - 1)], sent_emb[1:self.batch_size])
                    errD_mismatch = torch.nn.ReLU()(1.0 + output).mean()
                    # Fake images
                    noise = torch.randn(self.batch_size, cfg.GAN.Z_DIM)
                    if cfg.CUDA:
                        noise = noise.cuda()
                    fake_imgs = self.netG(noise, sent_emb)
                    fake_features = self.netD(fake_imgs.detach())
                    errD_fake = self.netD.cond_net(fake_features, sent_emb)
                    errD_fake = torch.nn.ReLU()(1.0 + errD_fake).mean()
                    # Update discriminator
                    errD = errD_real + (errD_fake + errD_mismatch) / 2.0
                    optimizerD.zero_grad()
                    optimizerG.zero_grad()
                    errD.backward()
                    optimizerD.step()
                    # Matching Aware Gradient Penalty (MA-GP)
                    interpolated = (imgs.data).requires_grad_()
                    sent_inter = (sent_emb.data).requires_grad_()
                    features = self.netD(interpolated)
                    out = self.netD.cond_net(features, sent_inter)
                    grads = torch.autograd.grad(outputs=out,
                                                inputs=(interpolated, sent_inter),
                                                grad_outputs=torch.ones(out.size()).cuda(),
                                                retain_graph=True,
                                                create_graph=True,
                                                only_inputs=True)
                    grad0 = grads[0].view(grads[0].size(0), -1)
                    grad1 = grads[1].view(grads[1].size(0), -1)
                    grad = torch.cat([grad0, grad1], dim=1)
                    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                    d_loss_gp = torch.mean((grad_l2norm) ** 6)
                    d_loss = 2.0 * d_loss_gp
                    optimizerD.zero_grad()
                    optimizerG.zero_grad()
                    d_loss.backward()
                    optimizerD.step()
                    # Update generator
                    fake_features = self.netD(fake_imgs)
                    output = self.netD.cond_net(fake_features, sent_emb)
                    errG = -output.mean()
                    optimizerG.zero_grad()
                    optimizerD.zero_grad()
                    errG.backward()
                    optimizerG.step()
                    # Log step
                    if step % 100 == 0:
                        print("[{}/{}][{}/{}] Loss_D: {:.3f} Loss_G: {:.3f} ({:.2f}h)".format(epoch,
                                                                                              cfg.TRAIN.MAX_EPOCH,
                                                                                              step,
                                                                                              len(self.train_dataloader),
                                                                                              errD.item(),
                                                                                              errG.item(),
                                                                                              (time.time() - start_time) / 3600))
                        logging.info("[{}/{}][{}/{}] Loss_D: {:.3f} Loss_G: {:.3f} ({:.2f}h)".format(epoch,
                                                                                              cfg.TRAIN.MAX_EPOCH,
                                                                                              step,
                                                                                              len(self.train_dataloader),
                                                                                              errD.item(),
                                                                                              errG.item(),
                                                                                              (time.time() - start_time) / 3600))

                # Save images
                if not os.path.exists(self.image_dir):
                    os.makedirs(self.image_dir)
                vutils.save_image(fake_imgs.data, '%s/fake_samples_epoch_%03d.png' % (self.image_dir, epoch),
                                  normalize=True)
                if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:

                    # Save model
                    self.save_model(epoch)
            self.save_model(cfg.TRAIN.MAX_EPOCH)
        else:
            self.gan_trainer.train()
        
        #################################################

    def build_models(self):
        self.text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TRAIN.RNN_ENCODER, map_location=lambda storage, loc: storage)
        self.text_encoder.load_state_dict(state_dict)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()
        self.netG = GENERATOR()
        self.netD = DISCRIMINATOR()
        if cfg.TRAIN.GENERATOR != '':
            state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.GENERATOR),
                                    map_location=lambda storage, loc: storage)
            self.netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.GENERATOR)
            logging.info('Load G from: '+ cfg.TRAIN.GENERATOR)
            # logging.info('Load G from: ' + cfg.TRAIN.GENERATOR)
            # istart = cfg.TRAIN.GENERATOR.rfind('_') + 1
            # iend = cfg.TRAIN.GENERATOR.rfind('.')
            # epoch = cfg.TRAIN.NET_G[istart:iend]
            # epoch = int(epoch) + 1
            Dname = os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.DISCRIMINATOR)
            state_dict = torch.load(Dname, map_location=lambda storage, loc: storage)
            self.netD.load_state_dict(state_dict)
        for p in self.netG.parameters():
            p.requires_grad = False
        print('Load generator from:', cfg.TRAIN.GENERATOR)
        logging.info('Load generator from:' + cfg.TRAIN.GENERATOR)

        self.start_epoch = 0
        if cfg.CUDA:
            self.text_encoder = self.text_encoder.to(self.device)
            self.netG = self.netG.to(self.device)
            self.netD = self.netD.to(self.device)

    def get_optimizers(self):
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=cfg.TRAIN.GENERATOR_LR, betas=(0.0, 0.9))
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.0, 0.9))
        return optimizerG, optimizerD
    
    def generate_eval_data(self):
        if cfg.GAN.TYPE == 'MY_GAN':
            # load the text encoder model to generate images for evaluation
            self.text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = torch.load(cfg.TRAIN.RNN_ENCODER, map_location=lambda storage, loc: storage)
            self.text_encoder.load_state_dict(state_dict)
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            print('Load text encoder from:', cfg.TRAIN.RNN_ENCODER)
            logging.info('Load text encoder from:' + cfg.TRAIN.RNN_ENCODER)
            self.text_encoder.eval()

            # load the generator model to generate images for evaluation
            self.netG = GENERATOR()
            
            state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.GENERATOR), map_location=lambda storage, loc: storage)
            self.netG.load_state_dict(state_dict)
            for p in self.netG.parameters():
                p.requires_grad = False
            print('Load generator from:', cfg.TRAIN.GENERATOR)
            logging.info('Load generator from:' + cfg.TRAIN.GENERATOR)
            self.netG.eval()

            noise = Variable(torch.FloatTensor(self.batch_size, cfg.GAN.Z_DIM))

            if cfg.CUDA:
                self.text_encoder = self.text_encoder.to(self.device)
                self.netG = self.netG.to(self.device)
                noise = noise.to(self.device)

            for step, data in enumerate(self.test_dataloader, 0):
                imgs, captions, cap_lens, class_ids, keys, sent_idx = self.prepare_data(data)

                #################################################
                # TODO
                # word embedding might be returned as well
                # hidden = self.text_encoder.init_hidden(self.batch_size)
                # sent_emb = self.text_encoder(captions, cap_lens, hidden)
                # sent_emb = sent_emb.detach()
                #################################################

                noise.data.normal_(0, 1)
                hidden = self.text_encoder.init_hidden(self.batch_size)
                # cap_lens = 18
                words_embs, sent_emb = self.text_encoder(captions, cap_lens, hidden)
                #### check
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                # if words_embs.shape[2] == 16:
                # import ipdb;ipdb.set_trace()
                fake_imgs = self.netG(noise, sent_emb)

                #################################################
                # TODO
                # this part can be different, depending on which algorithm is used
                # the main purpose is generating synthetic images using caption embedding and latent vector (noise)
                # fake_img = self.netG(noise, sent_emb, ...)
                #################################################
                for j in range(self.batch_size):
                    if not os.path.exists(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j].split('/')[0])):
                        os.mkdir(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j].split('/')[0]))

                        im = fake_imgs[j].data.cpu().numpy()
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        print(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j] + '_{}.png'.format(sent_idx[j])))
                        logging.info(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j] + '_{}.png'.format(sent_idx[j])))
                        im.save(
                            os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j] + '_{}.png'.format(sent_idx[j])))

        elif cfg.GAN.TYPE == 'DM_GAN':
            # load the text encoder model to generate images for evaluation
            self.text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            # state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.RNN_ENCODER), map_location=lambda storage, loc: storage)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            self.text_encoder.load_state_dict(state_dict)
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            logging.info('Load text encoder from:' + cfg.TRAIN.NET_E)
            self.text_encoder.eval()

            # load the generator model to generate images for evaluation
            self.netG = G_NET() #GENERATOR()
            # state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.GENERATOR), map_location=lambda storage, loc: storage)
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            new_netG_dict = self.netG.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in new_netG_dict}
            new_netG_dict.update(pretrained_dict)
            self.netG.load_state_dict(new_netG_dict)
            for p in self.netG.parameters():
                p.requires_grad = False
            print('Load generator from: ', model_dir)
            logging.info('Load generator from: ' + model_dir)
            self.netG.eval()

            noise = Variable(torch.FloatTensor(self.batch_size, cfg.GAN.Z_DIM))

            if cfg.CUDA:
                self.text_encoder = self.text_encoder.to(self.device)
                self.netG = self.netG.to(self.device)
                noise = noise.to(self.device)

            for step, data in enumerate(self.test_dataloader, 0):
                imgs, captions, cap_lens, class_ids, keys, sent_idx = self.prepare_data(data)

                #################################################
                # TODO
                # word embedding might be returned as well
                # hidden = self.text_encoder.init_hidden(self.batch_size)
                # sent_emb = self.text_encoder(captions, cap_lens, hidden)
                # sent_emb = sent_emb.detach()
                #################################################

                noise.data.normal_(0, 1)
                hidden = self.text_encoder.init_hidden(self.batch_size)
                # cap_lens = 18
                words_embs, sent_emb = self.text_encoder(captions, cap_lens, hidden)
                mask = (captions == 0)
                # if words_embs.shape[2] == 16:
                # import ipdb;ipdb.set_trace()
                fake_imgs, attention_maps, _, _ = self.netG(noise, sent_emb, words_embs, mask, cap_lens)

                #################################################
                # TODO
                # this part can be different, depending on which algorithm is used
                # the main purpose is generating synthetic images using caption embedding and latent vector (noise)
                # fake_img = self.netG(noise, sent_emb, ...)
                #################################################
                for j in range(self.batch_size):
                    if not os.path.exists(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j].split('/')[0])):
                        os.mkdir(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j].split('/')[0]))

                    for k in range(len(fake_imgs)):
                        if k==2:
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            im = np.transpose(im, (1, 2, 0))
                            im = Image.fromarray(im)
                            # print(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j] + '_{}.png'.format(sent_idx[j])))
                            # im.save(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j] + '_{}.png'.format(sent_idx[j])))
                            print(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j] + '_{}.png'.format(sent_idx[j])))
                            im.save(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j] + '_{}.png'.format(sent_idx[j])))
                            logging.info(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j] + '_{}.png'.format(sent_idx[j])))

                    # for k in range(len(attention_maps)):
                    #     cap_lens_np = np.asarray(cap_lens)
                    #     im = fake_imgs[k+1].detach().cpu()
                    #     attn_maps = attention_maps[k]
                    #     att_sze = attn_maps.size(2)
                    #     img_set, sentences = \
                    #         build_super_images2(im[j].unsqueeze(0),
                    #                             captions[j].unsqueeze(0),
                    #                             [cap_lens_np[j]], self.test_ixtoword,
                    #                             [attn_maps[j]], att_sze)
                    #     if img_set is not None:
                    #         # import ipdb; ipdb.set_trace()
                    #         im = Image.fromarray(img_set)
                    #         print(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j] + '_{}_a.png'.format(sent_idx[j])))
                    #         im.save(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j] + '_{}_a.png'.format(sent_idx[j])))


    def save_model(self, epoch):
        """
        Saves models
        """
        self.model_dir = os.path.join(output_dir, 'Model')
        mkdir_p(self.model_dir)
        torch.save(self.netG.state_dict(), '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        torch.save(self.netD.state_dict(), '%s/netD_epoch_%d.pth' % (self.model_dir, epoch))
        # load_params(netG, backup_para)
        torch.save(self.text_encoder.state_dict(), os.path.join(self.model_dir, cfg.TRAIN.RNN_ENCODER))
        # torch.save(self.image_encoder.state_dict(), os.path.join(self.output_dir, cfg.TRAIN.CNN_ENCODER))
        print('Save G/Ds models.')
        logging.info('Save G/Ds models.')
        # logging.info('Save G/Ds models.')
        # torch.save(self.netG.state_dict(), os.path.join(self.output_dir, cfg.TRAIN.GENERATOR))
        # torch.save(self.text_encoder.state_dict(), os.path.join(self.output_dir, cfg.TRAIN.RNN_ENCODER))
        # torch.save(self.image_encoder.state_dict(), os.path.join(self.output_dir, cfg.TRAIN.CNN_ENCODER))
