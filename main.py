import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import os, nltk
import numpy as np

from miscc.config import cfg, cfg_from_file
import pprint
import datetime
import dateutil.tz

from utils.data_utils import CUBDataset
from utils.trainer import trainer
import horovod.torch as hvd
# import sys
# old_stdout = sys.stdout

# Set a config file as 'train_birds.yml' in training, as 'eval_birds.yml' for evaluation
cfg_from_file('cfg/train_birds_mygan.yml') # eval_birds.yml # train_birds.yml #train_birds_dfgan.yml

print('Using config:')
pprint.pprint(cfg)

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
version_name = cfg.GAN.TYPE+cfg.VERSION_NAME
version_name += '_tree{}'.format(cfg.TREE.BRANCH_NUM)
if cfg.GAN.TYPE == 'MY_GAN':
    version_name += cfg.TRAIN.RNN_ENCODER.split('/')[2]
else:
    version_name += cfg.TRAIN.NET_E.split('/')[2]
if cfg.CONTRASTIVE.IMAGE_CONTRASTIVE:
    version_name += '_contrastive'
output_dir = 'sample/%s_%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, version_name, timestamp)

# log_file = open(os.path.join(output_dir, "/message.log"), "w")
# sys.stdout = log_file

imsize = cfg.TREE.BASE_SIZE * (4 ** (cfg.TREE.BRANCH_NUM - 1))
image_transform = transforms.Compose([
    transforms.Resize(int(imsize)),
    transforms.RandomCrop(imsize),
    transforms.RandomHorizontalFlip()])

train_dataset = CUBDataset(cfg.DATA_DIR, transform=image_transform, split='train')
test_dataset = CUBDataset(cfg.DATA_DIR, transform=image_transform, split='test')

print(f'train data directory:\n{train_dataset.split_dir}')
print(f'test data directory:\n{test_dataset.split_dir}\n')

print(f'# of train filenames:{train_dataset.filenames.shape}')
print(f'# of test filenames:{test_dataset.filenames.shape}\n')

print(f'example of filename of train image:{train_dataset.filenames[0]}')
print(f'example of filename of valid image:{test_dataset.filenames[0]}\n')

print(f'example of caption and its ids:\n{train_dataset.captions[0]}\n{train_dataset.captions_ids[0]}\n')
print(f'example of caption and its ids:\n{test_dataset.captions[0]}\n{test_dataset.captions_ids[0]}\n')

print(f'# of train captions:{np.asarray(train_dataset.captions).shape}')
print(f'# of test captions:{np.asarray(test_dataset.captions).shape}\n')

print(f'# of train caption ids:{np.asarray(train_dataset.captions_ids).shape}')
print(f'# of test caption ids:{np.asarray(test_dataset.captions_ids).shape}\n')

# For multi-GPU training
# hvd.init()
# torch.cuda.set_device(hvd.local_rank())
# train_sampler = torch.utils.data.distributed.DistributedSampler(
#     train_dataset,
#     num_replicas=hvd.size(),
#     rank=hvd.rank())
# test_sampler = torch.utils.data.distributed.DistributedSampler(
#     test_dataset,
#     num_replicas=hvd.size(),
#     rank=hvd.rank())

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
        drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS)) #, sampler=train_sampler
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE,
        drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS)) #, sampler=test_sampler

## 2. Define models and go to train/evaluate

algo = trainer(output_dir, train_dataset, train_dataloader, test_dataset, test_dataloader, train_dataset.ixtoword, test_dataset.ixtoword)

if cfg.TRAIN.FLAG:
    algo.train()
else:
    algo.generate_eval_data()
## 3. Measure Inception score and R-precision of given test dataset

#After set the config file as 'eval_birds.yml' and run the 'algo.generate_eval_data()',
# the synthesized images based on given captions and set of image and caption features
# should be saved inside a 'evaluation' folder, specifically in 'evaluation/generated_images/..'.

#**Then, go to the 'evaluation' folder and run each 'inception_score.ipynb' and 'r_precision.ipynb' file
# in order to measure inception score and r-precision score.**
