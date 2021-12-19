import numpy as np
from six.moves import urllib
import os, sys, tarfile
import pandas as pd
import pickle
import nltk
from PIL import Image
import random
from miscc.config import cfg

from nltk.tokenize import RegexpTokenizer

import torch
import torchvision.transforms as transforms

def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    
    #################################################
    # TODO
    # this part can be different, depending on which method is used
    # Assert that ret[-1] is image with size 128x128
    # ret = [normalize(img)]
    for i in range(cfg.TREE.BRANCH_NUM):
        if i < (cfg.TREE.BRANCH_NUM):
            re_img = transforms.Scale(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))
    #################################################

    return ret

class CUBDataset():
    def __init__(self, data_dir, transform, split='train', imsize=128, eval_mode=False, for_wrong=False):
        # evaluate MP metric with 'eval_mode' (return both original image and manipulated image from evaluation folder)
        # evaluate R-precision metric with 'for_wrong' (load caption from 'test_correct')
        self.imsize = imsize
        self.split = split
        self.eval_mode = eval_mode
        self.for_wrong = for_wrong
        
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform = transform
        self.imsize = []
        base_size = cfg.TREE.BASE_SIZE
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size*2)
            base_size = base_size * 2        

        self.current_dir = os.getcwd()
        print(f'self.current_dir:\n{self.current_dir}\n')
        if self.current_dir.split('/')[-1] == 'evaluation':
            self.current_dir = self.current_dir.replace('/evaluation', '')
        self.data_dir = os.path.join(self.current_dir, data_dir)
        print(f'self.data_dir:\n{self.data_dir}\n')

        self.image_dir = os.path.join(self.data_dir, 'CUB-200-2011/images')
        print(f'self.image_dir:\n{self.image_dir}\n')
        
        self.bbox = self.load_bbox()
        
        self.split_dir = os.path.join(self.data_dir, split)
        
        self.filenames, self.captions, self.captions_ids, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(self.data_dir, split)
        self.class_id = self.load_class_id(self.split_dir)
            
    def load_bbox(self):
        """
        Loads the corresponding bounded box of each bird image, which is saved in a txt format.
        
        Outputs:
        - filename_bbox: directionary, e.g., {'001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111': [x-left, y-top, width, height], .....}
        """
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)

        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()

        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox

        return filename_bbox
    
    def load_filenames(self, data_dir, split):
        """
        Loads assigned filenames for 'train' and 'test'.
        
        Inputs:
        - data_dir: base directory of dataset. ('data/birds')
        - split: either 'train' or 'test'

        Outputs:
        - filenames: numpy array, e.g., ['001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111', '001.Black_footed_Albatross/Black_Footed_Albatross_0002_55', ....]
        """
        filepath = os.path.join(data_dir, split, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f, encoding='latin1')
        #print(f'Load filenames:\t{filepath}')
        filenames = np.asarray(filenames)
        return filenames
    
    def load_class_id(self, data_dir):
        """
        Loads class ids of each image file.

        Inputs:
        - data_dir: directroy of either 'train' or 'test' dataset. ('data/birds/train' or 'data/birds/test')

        Outputs:
        - class_id: list, e.g., [1, 1, 1, 1, 1, ..... 2, 2, 2, ....]
          1 means '001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111', 2 means '002.Laysan_Albatross/Laysan_Albatross_0001_545'
        """
        with open(data_dir + '/class_info.pickle', 'rb') as f:
            class_id = pickle.load(f, encoding='latin1')

        return class_id
   
    def load_captions(self, data_dir, filenames, for_wrong=False):
        """
        Loads 10 captions for each image of either 'train' or 'test' dataset and tokenizes each caption as words.

        Inputs:
        - data_dir: base directory of dataset ('data/birds')
        - filenames: filenames of either 'train' or 'test' dataset

        Outputs:
        - all_captions: list
        """
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text_c10/train/%s.txt' % (data_dir, filenames[i])
            if not os.path.exists(cap_path):
                if for_wrong:
                    cap_path = '%s/text_c10/test_correct/%s.txt' % (data_dir, filenames[i])
                else:
                    cap_path = '%s/text_c10/test/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        #print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == cfg.TEXT.CAPTIONS_PER_IMAGE:
                        break
                if cnt < cfg.TEXT.CAPTIONS_PER_IMAGE:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions
    
    def make_new_captions(self, wordtoix, train_captions, test_captions):
        """
        Loads both train and test captions and generate vocab. Based on the vocab, assigns id to each word.

        Inputs:
        - wordtoix: dictionary
        - train_captions: list
        - test_captions: list

        Outputs:
        - train_captions_new: list, Ids of tokenized words of train captions
        - test_captions_new: list, Ids of tokenized words of test captions
        """

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new]
    
    def load_text_data(self, data_dir, split):
        """
        Computes the backward pass for a convolutional layer.

        Inputs:
        - data_dir: base directory of dataset ('data/birds')
        - split: either 'train' or 'test'

        Outputs:
        - filenames: list, filenames of either 'train' or 'test' datset depending on 'split'
        - captions: list, tokenized words of either 'train' or 'test' datset
        - captions_ids: list, ids of tokenized words of either 'train' or 'test' datset
        - ixtoword: dictionary, id to word based on generated vocab
        - wordtoix: dictionary, word to id based on generated vocab
        - n_words: scalar, length of generated vocab
        """
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        train_captions = self.load_captions(data_dir, train_names)
        test_captions = self.load_captions(data_dir, test_names, for_wrong=self.for_wrong)

        with open(filepath, 'rb') as f:
            print("filepath", filepath)
            x = pickle.load(f)
            ixtoword, wordtoix = x[2], x[3]
            del x
            n_words = len(ixtoword)
            print('Load from: ', filepath)

        train_captions_ids, test_captions_ids = self.make_new_captions(wordtoix, train_captions, test_captions)

        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            captions_ids = train_captions_ids
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            captions_ids = test_captions_ids
            filenames = test_names
        return filenames, captions, captions_ids, ixtoword, wordtoix, n_words

    def get_caption(self, sent_ix):
        """
        Modifies given caption by padding or shortening.

        Inputs:
        - sent_ix: randomly selected index

        Outputs:
        - x: np.array, padded ids of tokenized words or reduced ids by selecting frequent words
        - x_len: scalar, length of each tokenized caption
        """
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions_ids[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def get_mis_captions(self, img_cls_id, cap_cls_id):
        mis_match_captions_t = []
        mis_match_captions = torch.zeros(cfg.WRONG_CAPTION, cfg.TEXT.WORDS_NUM)
        mis_match_captions_len = torch.zeros(cfg.WRONG_CAPTION)
        i = 0
        while len(mis_match_captions_t) < cfg.WRONG_CAPTION:
            idx = random.randint(0, len(self.filenames) - 1)
            if img_cls_id == self.class_id[idx] or cap_cls_id == self.class_id[idx]:
                continue
            # student modified
            sent_ix = random.randint(0, cfg.TEXT.CAPTIONS_PER_IMAGE - 1) # if not -1, new_sent_idx can become 29330: invalid idx
            new_sent_ix = idx * cfg.TEXT.CAPTIONS_PER_IMAGE + sent_ix
            caps_t, cap_len_t = self.get_caption(new_sent_ix)
            mis_match_captions_t.append(torch.from_numpy(caps_t).squeeze())
            mis_match_captions_len[i] = cap_len_t
            i = i + 1
        sorted_cap_lens, sorted_cap_indices = torch.sort(mis_match_captions_len, 0, True)

        for i in range(cfg.WRONG_CAPTION):
            mis_match_captions[i, :] = mis_match_captions_t[sorted_cap_indices[i]]

        return mis_match_captions.type(torch.LongTensor), sorted_cap_lens.type(torch.LongTensor)
    

    def __getitem__(self, index):
        if self.split is 'train':
            key = self.filenames[index] # 002.Laysan_Albatross/Laysan_Albatross_0002_1027
            cls_id = self.class_id[index]
            bbox = self.bbox[key] if self.bbox is not None else None # bounding box
            data_dir = '%s/CUB_200_2011' % self.data_dir

            img_name = '%s/images/%s.jpg' % (data_dir, key)
            imgs = get_imgs(img_name, self.imsize, bbox, self.transform, normalize=self.norm)
            sent_ix = random.randint(0, cfg.TEXT.CAPTIONS_PER_IMAGE-1) # caption index
            new_sent_ix = index * cfg.TEXT.CAPTIONS_PER_IMAGE + sent_ix
            caps, cap_len = self.get_caption(new_sent_ix)

            data = {'img': imgs, 'caps': caps, 'cap_len': cap_len, 'cls_id': cls_id, 'key': key, 'sent_ix': sent_ix}

            #################################################
            # TODO
            # this part can be different, depending on which method is used
            # data[''] = ...
            #################################################

        else:
            key = self.filenames[index//cfg.TEXT.CAPTIONS_PER_IMAGE] # 002.Laysan_Albatross/Laysan_Albatross_0002_1027
            cls_id = self.class_id[index//cfg.TEXT.CAPTIONS_PER_IMAGE]
            sent_ix = index % cfg.TEXT.CAPTIONS_PER_IMAGE # caption index
            bbox = self.bbox[key] if self.bbox is not None else None  # bounding box
            data_dir = '%s/CUB_200_2011' % self.data_dir

            caps, cap_len = self.get_caption(index)

            data = {'caps': caps, 'cap_len': cap_len, 'cls_id': cls_id, 'key': key, 'sent_ix': sent_ix, 'cap_ix': index}

            if self.eval_mode:
                gen_img_name = os.path.join(self.current_dir, cfg.TEST.GENERATED_TEST_IMAGES, '{}_{}.png'.format(key, sent_ix))
                gen_imgs = get_imgs(gen_img_name, self.imsize, bbox=None, transform=self.transform, normalize=self.norm)
                data['gen_img'] = gen_imgs

        return data


    def __len__(self):
        '''
        In training, random index will be selected within # of filenames (e.g., 8855)
        In evaluation, random index will be selected within # of captions (e.g., 29330 (2933*10) (since # of caption per image is 10))
        '''
        if self.split is 'train':
            return len(self.filenames)
        else:
            return len(self.captions)
