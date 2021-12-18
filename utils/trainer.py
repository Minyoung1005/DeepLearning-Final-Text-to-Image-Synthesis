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
#########################################

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
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        
        self.batch_size = cfg.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        
        self.n_words = train_dataset.n_words # size of the dictionary
        self.train_ixtoword = train_ixtoword
        self.test_ixtoword = test_ixtoword
        self.gan_trainer = condGANTrainer(output_dir, train_dataloader,self.n_words, self.train_ixtoword, train_dataset)
        self.gan_test = condGANTrainer(output_dir, test_dataloader, self.n_words, self.test_ixtoword, test_dataset)
        '''
        '''
        self.device = 'cuda:' + str(0) if torch.cuda.device_count() > 0 else 'cpu'
    
    def prepare_data(self, data):
        """
        Prepares data given by dataloader
        e.g., x = Variable(x).cuda()
        """
        imgs = data['img']
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

        real_imgs = []
        for i in range(len(imgs)):
            imgs[i] = imgs[i][sorted_cap_indices]
            if cfg.CUDA:
                real_imgs.append(Variable(imgs[i]).to(self.device))
            else:
                real_imgs.append(Variable(imgs[i]))

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
        self.gan_trainer.train()
        
        #################################################
        
    
    def generate_eval_data(self):
        # load the text encoder model to generate images for evaluation
        self.text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.RNN_ENCODER), map_location=lambda storage, loc: storage)
        self.text_encoder.load_state_dict(state_dict)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.RNN_ENCODER)
        self.text_encoder.eval()
        
        # load the generator model to generate images for evaluation
        self.netG = GENERATOR()
        state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.GENERATOR), map_location=lambda storage, loc: storage)
        self.netG.load_state_dict(state_dict)
        for p in self.netG.parameters():
            p.requires_grad = False
        print('Load generator from:', cfg.TRAIN.GENERATOR)
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
                im.save(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j] + '_{}.png'.format(sent_idx[j])))
        
        
    def save_model(self):
        """
        Saves models
        """
        torch.save(self.netG.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.GENERATOR))
        torch.save(self.text_encoder.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.RNN_ENCODER))
        torch.save(self.image_encoder.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.CNN_ENCODER))
